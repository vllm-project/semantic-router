package modelservice

import (
	"math"
	"strconv"
)

// probabilitySumTolerance bounds how far a returned distribution may sum from
// one, the same bound an http_classify distribution gets.
const probabilitySumTolerance = 0.02

// checkAnswers marks every answer that does not fit the question it answers as
// invalid_model_output, so neither the decision signal nor the selector acts on
// it. The built-in runtime cannot produce such an answer, but an engine
// attached through a deployment's endpoint can.
func checkAnswers(request Request, response Response) Response {
	for _, question := range request.Questions {
		answer, ok := response.Answers[question.ID]
		if !ok || answer.Error != "" || answerFits(question, answer) {
			continue
		}
		response.Answers[question.ID] = Answer{Type: answer.Type, Error: "invalid_model_output"}
	}
	return response
}

// answerFits reports whether an answer is of the type asked and answers the
// question. A Choice gives every declared option a probability and no other
// option, and chooses a most probable one. A Noul is a probability. A Score
// lies within its levels and comes with a distribution over them. Answer types
// the router does not ask for pass unchecked.
func answerFits(question Question, answer Answer) bool {
	if answer.Type != question.Type {
		return false
	}
	switch question.Type {
	case "choice":
		keys := make([]string, len(question.Choices))
		for index, choice := range question.Choices {
			keys[index] = choice.Key
		}
		return distributionOver(keys, answer.Probabilities) &&
			(answer.Choice == "" || mostProbable(answer.Choice, answer.Probabilities))
	case "noul":
		return answer.Noul >= 0 && answer.Noul <= 1
	case "score":
		keys := make([]string, len(question.Levels))
		for index := range question.Levels {
			keys[index] = strconv.Itoa(index)
		}
		return distributionOver(keys, answer.Probabilities) &&
			answer.Score >= 0 && answer.Score <= float64(len(keys)-1)
	}
	return true
}

// distributionOver reports whether probabilities cover exactly keys, each in
// [0, 1], with a total within the tolerance of one.
func distributionOver(keys []string, probabilities map[string]float64) bool {
	if len(probabilities) != len(keys) {
		return false
	}
	total := 0.0
	for _, key := range keys {
		probability, ok := probabilities[key]
		if !ok || probability < 0 || probability > 1 {
			return false
		}
		total += probability
	}
	return math.Abs(total-1) <= probabilitySumTolerance
}

// mostProbable reports whether choice has the largest probability, ties
// included.
func mostProbable(choice string, probabilities map[string]float64) bool {
	chosen, ok := probabilities[choice]
	if !ok {
		return false
	}
	for _, probability := range probabilities {
		if probability > chosen {
			return false
		}
	}
	return true
}
