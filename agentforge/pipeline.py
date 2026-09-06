"""The end-to-end AgentForge generation, execution, and improvement pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from .llm import LLMError, OpenAICompatibleClient


class PipelineError(RuntimeError):
    """Raised for invalid input or a failed pipeline stage."""


class LLMClient(Protocol):
    def complete_json(
        self, system_prompt: str, user_prompt: str, *, temperature: float = 0.2
    ) -> dict[str, Any]:
        ...


@dataclass
class AgentSpec:
    name: str
    purpose: str
    system_prompt: str
    capabilities: list[str]
    constraints: list[str]
    success_criteria: list[str]


@dataclass
class Evaluation:
    passed: bool
    score: float
    failures: list[str] = field(default_factory=list)
    feedback: str = ""


@dataclass
class PipelineResult:
    objective: str
    criteria: list[str]
    agent_spec: AgentSpec
    output: str
    evaluation: Evaluation
    improved_spec: AgentSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentForgePipeline:
    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or OpenAICompatibleClient()

    def run(self, objective: str, criteria: list[str]) -> PipelineResult:
        objective = objective.strip()
        criteria = [item.strip() for item in criteria if item.strip()]
        if not objective:
            raise PipelineError("An objective is required.")
        if not criteria:
            raise PipelineError("At least one evaluation criterion is required.")

        try:
            spec = self._generate_spec(objective, criteria)
            output = self._execute(spec, objective)
            evaluation = self._evaluate(output, objective, criteria)
            improved = (
                self._improve(spec, objective, criteria, output, evaluation)
                if not evaluation.passed
                else None
            )
        except LLMError as exc:
            raise PipelineError(str(exc)) from exc
        return PipelineResult(
            objective=objective,
            criteria=criteria,
            agent_spec=spec,
            output=output,
            evaluation=evaluation,
            improved_spec=improved,
        )

    def _generate_spec(self, objective: str, criteria: list[str]) -> AgentSpec:
        data = self.client.complete_json(
            "You design reliable specialized AI agents. Return only valid JSON.",
            json.dumps(
                {
                    "task": "Create an agent specification for this objective.",
                    "objective": objective,
                    "success_criteria": criteria,
                    "required_fields": [
                        "name",
                        "purpose",
                        "system_prompt",
                        "capabilities",
                        "constraints",
                        "success_criteria",
                    ],
                }
            ),
        )
        return self._spec_from(data, criteria)

    def _execute(self, spec: AgentSpec, objective: str) -> str:
        data = self.client.complete_json(
            "You execute a specialized agent. Return only valid JSON.",
            json.dumps(
                {
                    "agent_spec": asdict(spec),
                    "objective": objective,
                    "required_fields": ["output"],
                }
            ),
            temperature=0.4,
        )
        output = data.get("output")
        if not isinstance(output, str) or not output.strip():
            raise PipelineError("The generated agent returned no output.")
        return output.strip()

    def _evaluate(
        self, output: str, objective: str, criteria: list[str]
    ) -> Evaluation:
        data = self.client.complete_json(
            "You are a rigorous evaluator. Return only valid JSON.",
            json.dumps(
                {
                    "objective": objective,
                    "criteria": criteria,
                    "agent_output": output,
                    "required_fields": [
                        "passed",
                        "score",
                        "failures",
                        "feedback",
                    ],
                }
            ),
        )
        passed = data.get("passed")
        score = data.get("score")
        failures = data.get("failures", [])
        feedback = data.get("feedback", "")
        if not isinstance(passed, bool) or not isinstance(score, (int, float)):
            raise PipelineError("The evaluator returned an invalid result.")
        if not isinstance(failures, list) or not all(
            isinstance(item, str) for item in failures
        ):
            raise PipelineError("The evaluator returned invalid failure details.")
        if not isinstance(feedback, str):
            raise PipelineError("The evaluator returned invalid feedback.")
        return Evaluation(
            passed=passed,
            score=max(0.0, min(1.0, float(score))),
            failures=failures,
            feedback=feedback,
        )

    def _improve(
        self,
        spec: AgentSpec,
        objective: str,
        criteria: list[str],
        output: str,
        evaluation: Evaluation,
    ) -> AgentSpec:
        data = self.client.complete_json(
            "You improve agent specifications based on concrete evaluation failures. "
            "Return only valid JSON.",
            json.dumps(
                {
                    "current_spec": asdict(spec),
                    "objective": objective,
                    "criteria": criteria,
                    "agent_output": output,
                    "evaluation": asdict(evaluation),
                    "required_fields": [
                        "name",
                        "purpose",
                        "system_prompt",
                        "capabilities",
                        "constraints",
                        "success_criteria",
                    ],
                }
            ),
        )
        return self._spec_from(data, criteria)

    @staticmethod
    def _spec_from(data: dict[str, Any], fallback_criteria: list[str]) -> AgentSpec:
        fields = ("name", "purpose", "system_prompt")
        if any(not isinstance(data.get(field), str) or not data[field].strip() for field in fields):
            raise PipelineError("The LLM returned an incomplete agent specification.")
        list_fields = ("capabilities", "constraints", "success_criteria")
        values: dict[str, list[str]] = {}
        for field in list_fields:
            value = data.get(field, fallback_criteria if field == "success_criteria" else [])
            if not isinstance(value, list) or not all(
                isinstance(item, str) and item.strip() for item in value
            ):
                raise PipelineError(f"The LLM returned invalid '{field}'.")
            values[field] = [item.strip() for item in value]
        return AgentSpec(
            name=data["name"].strip(),
            purpose=data["purpose"].strip(),
            system_prompt=data["system_prompt"].strip(),
            **values,
        )
