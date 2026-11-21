// Framing retained from the Claude plugin at d7ab7f587; the HTTP contract is shared.
export const ROUTE_QUESTIONS = {
  complexity: {
    type: "score",
    instructions: "How difficult is this request to solve correctly?",
    criteria: [
      "straightforward retrieval or formatting",
      "several steps of reasoning",
      "advanced reasoning or specialized expertise",
    ],
  },
  steps: {
    type: "score",
    instructions: "How many reasoning steps does this task need?",
    criteria: [
      "one or two steps",
      "several steps that build on each other",
      "many steps or long careful reasoning",
    ],
  },
};
export const FEEDBACK_QUESTION = {
  user_feedback: {
    type: "choice",
    instructions:
      "What feedback does this user turn give about the previous answer?",
    require_full_input: true,
    choices: [
      {
        key: "SAT",
        description: "the user is satisfied with the previous answer",
      },
      {
        key: "NEED_CLARIFICATION",
        description:
          "the user asks for clarification or a more detailed explanation",
      },
      {
        key: "WRONG_ANSWER",
        description:
          "the user says the previous answer was wrong or did not work",
      },
      {
        key: "WANT_DIFFERENT",
        description:
          "the user wants a different answer, format, style or approach",
      },
      {
        key: "NO_FEEDBACK",
        description: "the user gives no feedback on the previous answer",
      },
    ],
  },
};
export const REASK_QUESTION = {
  reask: {
    type: "noul",
    instructions:
      "Does the current user request seek the same information or action as the prior user request, including paraphrases and retries that add feedback? A different topic, objective or requested operation is not a repeat.",
  },
};
