import pytest
from src.graph.workflow import StepType, WorkflowStep, WorkflowTransition, WorkflowPlan


class TestWorkflowStep:
    def test_action_step(self):
        step = WorkflowStep(
            id="s1", name="Explore", instructions="Look at files",
            step_type=StepType.ACTION,
        )
        assert step.step_type == StepType.ACTION
        assert step.subworkflow_skill is None

    def test_subworkflow_step(self):
        step = WorkflowStep(
            id="s2", name="Invoke plans", instructions="",
            step_type=StepType.SUBWORKFLOW, subworkflow_skill="writing-plans",
        )
        assert step.subworkflow_skill == "writing-plans"


class TestWorkflowTransition:
    def test_unconditional(self):
        t = WorkflowTransition(from_step="a", to_step="b")
        assert t.condition is None

    def test_conditional(self):
        t = WorkflowTransition(from_step="a", to_step="b", condition="yes")
        assert t.condition == "yes"


class TestWorkflowPlan:
    def test_minimal_plan(self):
        step = WorkflowStep(id="main", name="Main", instructions="do it",
                            step_type=StepType.ACTION)
        plan = WorkflowPlan(name="test", steps=[step], transitions=[],
                            entry_step="main", constraints=[])
        assert plan.entry_step == "main"
        assert len(plan.steps) == 1

    def test_plan_with_constraints(self):
        step = WorkflowStep(id="s1", name="S1", instructions="",
                            step_type=StepType.ACTION)
        plan = WorkflowPlan(
            name="test", steps=[step], transitions=[],
            entry_step="s1", constraints=["No placeholders", "TDD always"],
        )
        assert len(plan.constraints) == 2
