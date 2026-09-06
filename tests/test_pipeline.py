import unittest

from agentforge.pipeline import AgentForgePipeline, PipelineError


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def complete_json(self, system_prompt, user_prompt, *, temperature=0.2):
        self.calls.append((system_prompt, user_prompt, temperature))
        return next(self.responses)


def spec(name="Researcher"):
    return {
        "name": name,
        "purpose": "Answer the objective",
        "system_prompt": "Be accurate and concise.",
        "capabilities": ["research"],
        "constraints": ["Do not invent facts"],
        "success_criteria": ["Accurate"],
    }


class PipelineTests(unittest.TestCase):
    def test_runs_and_improves_failed_agent(self):
        client = FakeClient(
            [
                spec(),
                {"output": "Initial answer"},
                {"passed": False, "score": 0.4, "failures": ["Missing sources"], "feedback": "Add sources."},
                spec("Improved Researcher"),
            ]
        )
        result = AgentForgePipeline(client).run("Research a topic", ["Accurate"])
        self.assertFalse(result.evaluation.passed)
        self.assertEqual(result.improved_spec.name, "Improved Researcher")
        self.assertEqual(len(client.calls), 4)

    def test_does_not_improve_passing_agent(self):
        client = FakeClient(
            [spec(), {"output": "Good answer"}, {"passed": True, "score": 1, "failures": [], "feedback": ""}]
        )
        result = AgentForgePipeline(client).run("Do a task", ["Useful"])
        self.assertIsNone(result.improved_spec)
        self.assertEqual(len(client.calls), 3)

    def test_rejects_missing_inputs(self):
        pipeline = AgentForgePipeline(FakeClient([]))
        with self.assertRaisesRegex(PipelineError, "objective"):
            pipeline.run("", ["criterion"])
        with self.assertRaisesRegex(PipelineError, "criterion"):
            pipeline.run("objective", [])

    def test_rejects_malformed_evaluation(self):
        client = FakeClient([spec(), {"output": "answer"}, {"passed": "yes", "score": 1}])
        with self.assertRaisesRegex(PipelineError, "evaluator"):
            AgentForgePipeline(client).run("objective", ["criterion"])


if __name__ == "__main__":
    unittest.main()
