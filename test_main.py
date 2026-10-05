import importlib.util
import io
import json
import logging
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import HTTPException

import main


class ReaderServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.url = "https://example.com/private?token=secret"
        self.client = AsyncMock()
        self.client.__aenter__.return_value = self.client
        self.client.__aexit__.return_value = None
        self.client_patch = patch("main.httpx.AsyncClient", return_value=self.client)
        self.client_patch.start()
        self.addCleanup(self.client_patch.stop)
        main._account_failure = None
        self.addCleanup(setattr, main, "_account_failure", None)
        tavily_pool_patch = patch("main._tavily_pool", main.TavilyKeyPool([]))
        tavily_pool_patch.start()
        self.addCleanup(tavily_pool_patch.stop)

    @staticmethod
    def _status_error(status_code: int) -> httpx.HTTPStatusError:
        request = httpx.Request("GET", "https://r.jina.ai/https://example.com")
        response = httpx.Response(status_code, request=request)
        return httpx.HTTPStatusError(
            f"上游返回 {status_code}", request=request, response=response
        )

    async def test_timeout_retries_once_and_returns_structured_error(self):
        self.client.get.side_effect = [
            httpx.ReadTimeout("第一次超时"),
            httpx.ReadTimeout("第二次超时"),
        ]

        with patch("main.asyncio.sleep", new=AsyncMock()) as sleep:
            with self.assertRaises(HTTPException) as raised:
                await main.read_url(self.url)

        self.assertEqual(raised.exception.status_code, 504)
        self.assertEqual(
            raised.exception.detail,
            {
                "kind": "timeout",
                "message": "Jina Reader 请求超时",
                "retryable": True,
                "upstream_status": None,
                "attempts": 2,
                "duration_ms": unittest.mock.ANY,
            },
        )
        self.assertEqual(self.client.get.await_count, 2)
        sleep.assert_awaited_once()

    async def test_upstream_auth_does_not_retry(self):
        for status_code in (401, 402, 403):
            with self.subTest(status_code=status_code):
                self.client.get.reset_mock()
                self.client.get.side_effect = self._status_error(status_code)

                with self.assertRaises(HTTPException) as raised:
                    await main.read_url(self.url)

                self.assertEqual(raised.exception.status_code, 502)
                self.assertEqual(raised.exception.detail["kind"], "upstream_auth")
                self.assertFalse(raised.exception.detail["retryable"])
                self.assertEqual(
                    raised.exception.detail["upstream_status"], status_code
                )
                self.assertEqual(raised.exception.detail["attempts"], 1)
                self.assertEqual(self.client.get.await_count, 1)

    async def test_account_failure_turns_health_degraded_until_next_success(self):
        self.assertEqual(await main.health(), {"status": "ok"})
        self.client.get.side_effect = self._status_error(402)

        with self.assertRaises(HTTPException):
            await main.read_url(self.url)

        degraded = await main.health()
        self.assertEqual(degraded.status_code, 503)
        body = json.loads(degraded.body)
        self.assertEqual(body["jina_account"]["upstream_status"], 402)
        self.assertEqual(body["jina_account"]["message"], "Jina Reader 账户余额不足")
        self.assertNotIn("secret", degraded.body.decode())

        request = httpx.Request("GET", "https://r.jina.ai/https://example.com")
        self.client.get.side_effect = None
        self.client.get.return_value = httpx.Response(
            200, request=request, text="Title: ok\n\nbody"
        )
        await main.read_url(self.url)
        self.assertEqual(await main.health(), {"status": "ok"})

    async def test_page_level_failures_do_not_degrade_health(self):
        for status_code in (400, 404, 429, 500):
            with self.subTest(status_code=status_code):
                self.client.get.side_effect = self._status_error(status_code)
                with (
                    patch("main.asyncio.sleep", new=AsyncMock()),
                    self.assertRaises(HTTPException),
                ):
                    await main.read_url(self.url)
                self.assertEqual(await main.health(), {"status": "ok"})

    async def test_rate_limit_retries_once(self):
        self.client.get.side_effect = [
            self._status_error(429),
            self._status_error(429),
        ]

        with patch("main.asyncio.sleep", new=AsyncMock()):
            with self.assertRaises(HTTPException) as raised:
                await main.read_url(self.url)

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.detail["kind"], "rate_limited")
        self.assertTrue(raised.exception.detail["retryable"])
        self.assertEqual(raised.exception.detail["upstream_status"], 429)
        self.assertEqual(raised.exception.detail["attempts"], 2)
        self.assertEqual(self.client.get.await_count, 2)

    async def test_upstream_5xx_retries_once(self):
        self.client.get.side_effect = [
            self._status_error(503),
            self._status_error(503),
        ]

        with patch("main.asyncio.sleep", new=AsyncMock()):
            with self.assertRaises(HTTPException) as raised:
                await main.read_url(self.url)

        self.assertEqual(raised.exception.status_code, 502)
        self.assertEqual(raised.exception.detail["kind"], "upstream_error")
        self.assertTrue(raised.exception.detail["retryable"])
        self.assertEqual(raised.exception.detail["upstream_status"], 503)
        self.assertEqual(raised.exception.detail["attempts"], 2)

    async def test_fast_request_error_retries_once(self):
        request = httpx.Request("GET", "https://r.jina.ai/https://example.com")
        self.client.get.side_effect = httpx.ConnectError(
            "连接失败 token=secret", request=request
        )

        with patch("main.asyncio.sleep", new=AsyncMock()) as sleep:
            with self.assertRaises(HTTPException) as raised:
                await main.read_url(self.url)

        self.assertEqual(raised.exception.status_code, 502)
        self.assertEqual(raised.exception.detail["kind"], "request_error")
        self.assertTrue(raised.exception.detail["retryable"])
        self.assertIsNone(raised.exception.detail["upstream_status"])
        self.assertEqual(raised.exception.detail["attempts"], 2)
        self.assertEqual(raised.exception.detail["message"], "Jina Reader 请求失败")
        self.assertNotIn("secret", str(raised.exception.detail))
        self.assertEqual(self.client.get.await_count, 2)
        sleep.assert_awaited_once()

    async def test_unknown_error_is_structured_and_does_not_leak_message(self):
        self.client.get.side_effect = RuntimeError(
            "意外失败 https://example.com/private?token=secret"
        )

        with self.assertLogs("reader_service", level=logging.WARNING) as logs:
            with self.assertRaises(HTTPException) as raised:
                await main.read_url(self.url)

        self.assertEqual(raised.exception.status_code, 502)
        self.assertEqual(raised.exception.detail["kind"], "request_error")
        self.assertEqual(raised.exception.detail["message"], "Jina Reader 请求失败")
        self.assertFalse(raised.exception.detail["retryable"])
        self.assertEqual(raised.exception.detail["attempts"], 1)
        self.assertNotIn("secret", str(raised.exception.detail))
        self.assertNotIn("secret", "\n".join(logs.output))
        self.assertEqual(self.client.get.await_count, 1)

    async def test_invalid_url_does_not_echo_input(self):
        invalid_url = "javascript:alert('secret')"

        with self.assertRaises(HTTPException) as raised:
            await main.read_url(invalid_url)

        self.assertEqual(raised.exception.status_code, 422)
        self.assertEqual(raised.exception.detail, "URL 格式无效")
        self.assertNotIn("secret", raised.exception.detail)
        self.client.get.assert_not_awaited()

    async def test_malformed_ipv6_returns_fixed_422_without_echo(self):
        invalid_url = "https://[::1/private?token=secret"

        with self.assertRaises(HTTPException) as raised:
            await main.read_url(invalid_url)

        self.assertEqual(raised.exception.status_code, 422)
        self.assertEqual(raised.exception.detail, "URL 格式无效")
        self.assertNotIn("secret", raised.exception.detail)
        self.client.get.assert_not_awaited()

    async def test_invalid_port_returns_fixed_422_without_upstream_request(self):
        invalid_url = "https://example.com:not-a-port/private?token=secret"

        with self.assertRaises(HTTPException) as raised:
            await main.read_url(invalid_url)

        self.assertEqual(raised.exception.status_code, 422)
        self.assertEqual(raised.exception.detail, "URL 格式无效")
        self.assertNotIn("secret", raised.exception.detail)
        self.client.get.assert_not_awaited()

    async def test_other_upstream_4xx_does_not_retry(self):
        self.client.get.side_effect = self._status_error(400)

        with self.assertRaises(HTTPException) as raised:
            await main.read_url(self.url)

        self.assertEqual(raised.exception.status_code, 502)
        self.assertEqual(raised.exception.detail["kind"], "upstream_error")
        self.assertFalse(raised.exception.detail["retryable"])
        self.assertEqual(raised.exception.detail["upstream_status"], 400)
        self.assertEqual(raised.exception.detail["attempts"], 1)
        self.assertEqual(self.client.get.await_count, 1)

    async def test_retry_success_reports_attempts(self):
        request = httpx.Request("GET", "https://r.jina.ai/https://example.com")
        success = httpx.Response(
            200,
            text="# 示例标题\n正文",
            headers={"x-title": "Example Title"},
            request=request,
        )
        self.client.get.side_effect = [httpx.ReadTimeout("超时"), success]

        with patch("main.asyncio.sleep", new=AsyncMock()):
            response = await main.read_url(self.url)

        self.assertEqual(response.attempts, 2)
        self.assertEqual(response.title, "Example Title")
        self.assertEqual(response.content, "# 示例标题\n正文")

    async def test_failure_log_does_not_contain_full_url_or_query(self):
        self.client.get.side_effect = self._status_error(401)

        with self.assertLogs("reader_service", level=logging.WARNING) as logs:
            with self.assertRaises(HTTPException):
                await main.read_url(self.url)

        joined = "\n".join(logs.output)
        self.assertIn("domain=example.com", joined)
        self.assertIn("kind=upstream_auth", joined)
        self.assertNotIn("private", joined)
        self.assertNotIn("secret", joined)

    def test_container_disables_access_log_and_keeps_port_8091(self):
        dockerfile = Path(__file__).with_name("Dockerfile").read_text()

        self.assertIn("EXPOSE 8091", dockerfile)
        self.assertIn('"--port", "8091", "--no-access-log"', dockerfile)

    def test_workflow_runs_explicit_tests_before_deploy(self):
        workflow = (
            Path(__file__).with_name(".github") / "workflows" / "deploy.yml"
        ).read_text()

        explicit_test_command = "python -m unittest -v test_main"
        self.assertIn(explicit_test_command, workflow)
        self.assertNotIn("python -m unittest discover", workflow)
        self.assertLess(
            workflow.index(explicit_test_command),
            workflow.index("docker build -t reader-service ."),
        )

    def test_workflow_runs_real_read_smoke_after_liveness_check(self):
        workflow = (
            Path(__file__).with_name(".github") / "workflows" / "deploy.yml"
        ).read_text()

        smoke_command = "python scripts/smoke_read.py"
        self.assertIn(smoke_command, workflow)
        self.assertLess(
            workflow.index("curl -sf http://localhost:8091/health"),
            workflow.index(smoke_command),
        )

    def test_smoke_script_validates_content_and_attempts_without_leaking_target(self):
        script_path = Path(__file__).with_name("scripts") / "smoke_read.py"
        self.assertTrue(script_path.is_file())
        spec = importlib.util.spec_from_file_location("reader_smoke_read", script_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        smoke_read = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(smoke_read)

        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self, _limit):
                return b'{"content":"ok","attempts":1}'

        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch.object(smoke_read.request, "urlopen", return_value=FakeResponse()),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            exit_code = smoke_read.main()

        self.assertEqual(exit_code, 0)
        self.assertIn("reader smoke passed", stdout.getvalue())
        self.assertNotIn(smoke_read.SMOKE_TARGET, stdout.getvalue())
        self.assertNotIn(smoke_read.SMOKE_TARGET, stderr.getvalue())

    def test_smoke_script_failure_only_outputs_safe_diagnostics(self):
        script_path = Path(__file__).with_name("scripts") / "smoke_read.py"
        self.assertTrue(script_path.is_file())
        spec = importlib.util.spec_from_file_location(
            "reader_smoke_read_failure", script_path
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        smoke_read = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(smoke_read)

        secret_target = "https://example.com/private?token=secret"
        upstream_error = smoke_read.error.HTTPError(
            secret_target,
            502,
            "JINA_API_KEY=secret",
            {},
            io.BytesIO(
                b'{"detail":{"kind":"upstream_auth","upstream_status":401,'
                b'"message":"JINA_API_KEY=secret"}}'
            ),
        )
        stderr = io.StringIO()
        with (
            patch.object(smoke_read, "SMOKE_TARGET", secret_target),
            patch.object(smoke_read.request, "urlopen", side_effect=upstream_error),
            redirect_stderr(stderr),
        ):
            exit_code = smoke_read.main()

        output = stderr.getvalue()
        self.assertEqual(exit_code, 1)
        self.assertIn("http_status=502", output)
        self.assertIn("kind=upstream_auth", output)
        self.assertIn("upstream_status=401", output)
        self.assertNotIn(secret_target, output)
        self.assertNotIn("secret", output)
        self.assertNotIn("JINA_API_KEY", output)

    def test_key_check_script_never_prints_keys(self):
        script_path = Path(__file__).with_name("scripts") / "check_tavily_keys.py"
        spec = importlib.util.spec_from_file_location("check_tavily_keys", script_path)
        check = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(check)
        request = httpx.Request("GET", check.TAVILY_USAGE_URL)
        responses = {
            "tvly-good": httpx.Response(
                200,
                json={"account": {"current_plan": "Researcher", "plan_usage": 6,
                                  "plan_limit": 1000}},
                request=request,
            ),
            "tvly-full": httpx.Response(
                200,
                json={"account": {"plan_usage": 1000, "plan_limit": 1000}},
                request=request,
            ),
            "tvly-bad": httpx.Response(401, request=request),
        }

        class FakeClient:
            def __init__(self, *_args, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def get(self, _url, headers):
                return responses[headers["Authorization"].removeprefix("Bearer ")]

        stdout = io.StringIO()
        with (
            patch.dict(
                "os.environ",
                {"TAVILY_API_KEYS": "tvly-good, tvly-full,tvly-bad,tvly-good"},
            ),
            patch.object(check.httpx, "Client", FakeClient),
            redirect_stdout(stdout),
        ):
            exit_code = check.main()

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("key#1 ok plan=Researcher usage=6/1000", output)
        self.assertIn("key#2 exhausted", output)
        self.assertIn("key#3 http_status=401", output)
        self.assertIn("key#4 duplicate", output)
        self.assertIn("usable=1/4", output)
        self.assertNotIn("tvly-", output)

    def test_container_ships_scripts_and_pool_secret(self):
        dockerfile = Path(__file__).with_name("Dockerfile").read_text()
        workflow = (
            Path(__file__).with_name(".github") / "workflows" / "deploy.yml"
        ).read_text()

        self.assertIn("COPY scripts/ scripts/", dockerfile)
        self.assertIn("-e TAVILY_API_KEYS=${{ secrets.TAVILY_API_KEYS }}", workflow)

    async def test_total_deadline_prevents_second_attempt_without_budget(self):
        self.client.get.side_effect = httpx.ReadTimeout("超时")

        with (
            patch("main._monotonic", side_effect=[0.0, 18.0, 18.0]),
            patch("main.asyncio.sleep", new=AsyncMock()) as sleep,
        ):
            with self.assertRaises(HTTPException) as raised:
                await main.read_url(self.url)

        self.assertEqual(raised.exception.detail["kind"], "timeout")
        self.assertEqual(raised.exception.detail["attempts"], 1)
        self.assertEqual(self.client.get.await_count, 1)
        sleep.assert_not_awaited()

    async def test_slow_failure_without_ten_second_retry_window_does_not_retry(self):
        self.client.get.side_effect = httpx.ReadTimeout("慢请求超时")

        with (
            patch("main._monotonic", side_effect=[0.0, 8.1, 8.1, 8.1, 8.1]),
            patch("main.asyncio.sleep", new=AsyncMock()) as sleep,
        ):
            with self.assertRaises(HTTPException) as raised:
                await main.read_url(self.url)

        self.assertEqual(raised.exception.detail["kind"], "timeout")
        self.assertEqual(raised.exception.detail["attempts"], 1)
        self.assertEqual(self.client.get.await_count, 1)
        sleep.assert_not_awaited()

    async def test_total_deadline_expired_during_backoff_does_not_start_second_attempt(
        self,
    ):
        request = httpx.Request("GET", "https://r.jina.ai/https://example.com")
        success = httpx.Response(200, text="不应被读取", request=request)
        self.client.get.side_effect = [httpx.ReadTimeout("第一次超时"), success]

        with (
            patch("main._monotonic", side_effect=[0.0, 7.8, 18.01, 18.02]),
            patch("main.asyncio.sleep", new=AsyncMock()) as sleep,
        ):
            with self.assertRaises(HTTPException) as raised:
                await main.read_url(self.url)

        self.assertEqual(raised.exception.status_code, 504)
        self.assertEqual(raised.exception.detail["kind"], "timeout")
        self.assertEqual(raised.exception.detail["attempts"], 1)
        self.assertEqual(self.client.get.await_count, 1)
        sleep.assert_awaited_once()

    async def test_success_completed_after_total_deadline_is_rejected(self):
        request = httpx.Request("GET", "https://r.jina.ai/https://example.com")
        self.client.get.return_value = httpx.Response(
            200, text="过期结果", request=request
        )

        with patch("main._monotonic", side_effect=[0.0, 18.01, 18.02, 18.03]):
            with self.assertRaises(HTTPException) as raised:
                await main.read_url(self.url)

        self.assertEqual(raised.exception.status_code, 504)
        self.assertEqual(raised.exception.detail["kind"], "timeout")
        self.assertEqual(raised.exception.detail["attempts"], 1)
        self.assertEqual(self.client.get.await_count, 1)

    @staticmethod
    def _tavily_response(results: list, status_code: int = 200) -> httpx.Response:
        request = httpx.Request("POST", main.TAVILY_EXTRACT_URL)
        return httpx.Response(
            status_code, json={"results": results, "failed_results": []}, request=request
        )

    @staticmethod
    def _jina_success(text: str = "# Jina 标题\n正文") -> httpx.Response:
        request = httpx.Request("GET", "https://r.jina.ai/https://example.com")
        return httpx.Response(200, text=text, request=request)

    async def test_tavily_success_skips_jina(self):
        self.client.post.return_value = self._tavily_response(
            [{"url": self.url, "title": "Tavily 标题", "raw_content": "正文内容"}]
        )

        with patch("main._tavily_pool", main.TavilyKeyPool(["tvly-test"])):
            response = await main.read_url(self.url)

        self.assertEqual(response.provider, "tavily")
        self.assertEqual(response.content, "正文内容")
        self.assertEqual(response.title, "Tavily 标题")
        self.assertEqual(response.attempts, 1)
        self.client.get.assert_not_awaited()
        call = self.client.post.await_args
        self.assertEqual(call.args[0], main.TAVILY_EXTRACT_URL)
        self.assertEqual(call.kwargs["headers"]["Authorization"], "Bearer tvly-test")
        self.assertEqual(call.kwargs["json"]["urls"], [self.url])
        self.assertEqual(call.kwargs["json"]["extract_depth"], "basic")

    async def test_tavily_without_title_uses_first_heading(self):
        self.client.post.return_value = self._tavily_response(
            [{"url": self.url, "raw_content": "# 页面标题\n正文"}]
        )

        with patch("main._tavily_pool", main.TavilyKeyPool(["tvly-test"])):
            response = await main.read_url(self.url)

        self.assertEqual(response.title, "页面标题")

    async def test_tavily_empty_content_falls_back_to_jina(self):
        for results in ([], [{"url": self.url, "raw_content": "  "}]):
            with self.subTest(results=results):
                self.client.post.reset_mock()
                self.client.get.reset_mock()
                self.client.post.return_value = self._tavily_response(results)
                self.client.get.return_value = self._jina_success()

                with (
                    patch("main._tavily_pool", main.TavilyKeyPool(["tvly-test"])),
                    self.assertLogs("reader_service", level=logging.WARNING) as logs,
                ):
                    response = await main.read_url(self.url)

                self.assertEqual(response.provider, "jina")
                self.assertEqual(response.title, "Jina 标题")
                self.assertEqual(response.attempts, 2)
                self.assertEqual(self.client.get.await_count, 1)
                joined = "\n".join(logs.output)
                self.assertIn("reader_tavily_fallback", joined)
                self.assertIn("kind=empty_content", joined)
                self.assertNotIn("secret", joined)

    async def test_tavily_errors_fall_back_to_jina(self):
        errors = {
            "timeout": httpx.ReadTimeout("超时"),
            "upstream_error": self._status_error(500),
            "request_error": httpx.ConnectError("连不上"),
        }
        for kind, exc in errors.items():
            with self.subTest(kind=kind):
                self.client.post.reset_mock()
                self.client.get.reset_mock()
                self.client.post.side_effect = exc
                self.client.get.return_value = self._jina_success()

                with (
                    patch("main._tavily_pool", main.TavilyKeyPool(["tvly-test"])),
                    self.assertLogs("reader_service", level=logging.WARNING) as logs,
                ):
                    response = await main.read_url(self.url)

                self.assertEqual(response.provider, "jina")
                self.assertIn(f"kind={kind}", "\n".join(logs.output))
                self.assertEqual(await main.health(), {"status": "ok"})

    async def test_jina_fallback_only_gets_remaining_total_budget(self):
        self.client.post.side_effect = httpx.ReadTimeout("超时")

        with (
            patch("main._tavily_pool", main.TavilyKeyPool(["tvly-test"])),
            patch("main._monotonic", side_effect=[0.0, 0.0, 0.0, 18.0, 18.0, 18.0]),
        ):
            with self.assertRaises(HTTPException) as raised:
                await main.read_url(self.url)

        self.assertEqual(raised.exception.status_code, 504)
        self.assertEqual(raised.exception.detail["kind"], "timeout")
        self.assertEqual(raised.exception.detail["attempts"], 1)
        self.client.get.assert_not_awaited()

    def _tavily_ok(self) -> httpx.Response:
        return self._tavily_response([{"url": self.url, "raw_content": "正文"}])

    def _used_keys(self) -> list[str]:
        return [
            call.kwargs["headers"]["Authorization"].removeprefix("Bearer ")
            for call in self.client.post.await_args_list
        ]

    async def test_pool_rotates_keys_across_requests(self):
        self.client.post.return_value = self._tavily_ok()
        pool = main.TavilyKeyPool(["k1", "k2", "k3"])

        with patch("main._tavily_pool", pool):
            for _ in range(4):
                await main.read_url(self.url)

        self.assertEqual(self._used_keys(), ["k1", "k2", "k3", "k1"])

    async def test_account_failure_switches_key_within_same_request(self):
        for status_code in (401, 429, 432, 433):
            with self.subTest(status_code=status_code):
                self.client.post.reset_mock()
                self.client.post.side_effect = [
                    self._status_error(status_code),
                    self._tavily_ok(),
                ]
                pool = main.TavilyKeyPool(["k1", "k2"])

                with patch("main._tavily_pool", pool):
                    response = await main.read_url(self.url)

                self.assertEqual(response.provider, "tavily")
                self.assertEqual(self._used_keys(), ["k1", "k2"])
                self.assertEqual(pool.available_count(), 1)
                self.client.get.assert_not_awaited()

    async def test_unavailable_key_is_skipped_until_cooldown_expires(self):
        pool = main.TavilyKeyPool(["k1", "k2"])
        self.client.post.side_effect = [self._status_error(429), self._tavily_ok()]

        with (
            patch("main._tavily_pool", pool),
            patch("main._wall_time", return_value=1000.0),
        ):
            await main.read_url(self.url)
            self.client.post.reset_mock()
            self.client.post.side_effect = None
            self.client.post.return_value = self._tavily_ok()
            await main.read_url(self.url)
            await main.read_url(self.url)
            self.assertEqual(self._used_keys(), ["k2", "k2"])

        self.client.post.reset_mock()
        with (
            patch("main._tavily_pool", pool),
            patch("main._wall_time", return_value=1061.0),
        ):
            await main.read_url(self.url)
            await main.read_url(self.url)
        self.assertEqual(sorted(self._used_keys()), ["k1", "k2"])

    async def test_invalid_key_stays_disabled(self):
        pool = main.TavilyKeyPool(["k1", "k2"])
        self.client.post.side_effect = [self._status_error(401), self._tavily_ok()]

        with (
            patch("main._tavily_pool", pool),
            patch("main._wall_time", return_value=1000.0),
        ):
            await main.read_url(self.url)

        self.client.post.reset_mock()
        self.client.post.side_effect = None
        self.client.post.return_value = self._tavily_ok()
        with (
            patch("main._tavily_pool", pool),
            patch("main._wall_time", return_value=10**9),
        ):
            for _ in range(3):
                await main.read_url(self.url)
        self.assertEqual(self._used_keys(), ["k2", "k2", "k2"])

    async def test_all_keys_unavailable_degrades_health_and_falls_back(self):
        pool = main.TavilyKeyPool(["k1", "k2"])
        self.client.post.side_effect = [self._status_error(432), self._status_error(401)]
        self.client.get.return_value = self._jina_success()

        with patch("main._tavily_pool", pool):
            response = await main.read_url(self.url)
            self.assertEqual(response.provider, "jina")
            degraded = await main.health()

            self.assertEqual(degraded.status_code, 503)
            body = json.loads(degraded.body)
            self.assertEqual(body["tavily_account"]["total"], 2)
            self.assertEqual(body["tavily_account"]["available"], 0)
            self.assertEqual(
                body["tavily_account"]["key_statuses"], {"1": 432, "2": 401}
            )
            self.assertNotIn("jina_account", body)
            self.assertNotIn("k1", degraded.body.decode())

            self.client.post.reset_mock()
            self.client.get.reset_mock()
            self.client.get.return_value = self._jina_success()
            with self.assertLogs("reader_service", level=logging.WARNING) as logs:
                response = await main.read_url(self.url)
            self.assertEqual(response.provider, "jina")
            self.client.post.assert_not_awaited()
            self.assertIn("kind=no_available_key", "\n".join(logs.output))

    async def test_page_level_tavily_failure_does_not_burn_other_keys(self):
        pool = main.TavilyKeyPool(["k1", "k2"])
        self.client.post.side_effect = self._status_error(500)
        self.client.get.return_value = self._jina_success()

        with patch("main._tavily_pool", pool):
            response = await main.read_url(self.url)

        self.assertEqual(response.provider, "jina")
        self.assertEqual(self._used_keys(), ["k1"])
        self.assertEqual(pool.available_count(), 2)

    async def test_client_uses_split_timeout_configuration(self):
        request = httpx.Request("GET", "https://r.jina.ai/https://example.com")
        self.client.get.return_value = httpx.Response(200, text="正文", request=request)

        await main.read_url(self.url)

        timeout = main.httpx.AsyncClient.call_args.kwargs["timeout"]
        self.assertEqual(timeout.connect, main.JINA_CONNECT_TIMEOUT)
        self.assertEqual(timeout.read, main.JINA_READ_TIMEOUT)
        self.assertEqual(timeout.write, main.JINA_WRITE_TIMEOUT)
        self.assertEqual(timeout.pool, main.JINA_POOL_TIMEOUT)
        self.assertGreaterEqual(timeout.read, 10)
        self.assertGreaterEqual(main.JINA_MIN_RETRY_WINDOW, 10)
        self.assertLess(main.JINA_TOTAL_TIMEOUT, 20)


if __name__ == "__main__":
    unittest.main()
