from collections import defaultdict
import math
import nltk
from decimal import *

# All the other sequences are unicode, might as well be consistent
START_SYMBOL = u'<S>'
END_SYMBOL = u'</S>'
LOW_PROB = Decimal(0.0000000001)
DELTA_LIMIT = Decimal(0.1)
getcontext().prec=256


def bigram_counters(tagged_sentences):
    # make all of these counters in one go, let's avoid looping over and over
    tag_counter = defaultdict(Decimal)
    word_tag_counter = defaultdict(Decimal)
    tag_previous_tag_counter = defaultdict(Decimal)
    for tagged_sentence in tagged_sentences:
        # Include the starting symbol in the tags
        tag_counter[START_SYMBOL] += 1
        previous_tag = START_SYMBOL
        for word, tag in tagged_sentence:
            word_tag_counter[word, tag] += 1
            tag_counter[tag] += 1
            tag_previous_tag_counter[tag, previous_tag] += 1
            previous_tag = tag
        # Now add the ending symbol to the tags
        tag_counter[END_SYMBOL] += 1
        tag_previous_tag_counter[END_SYMBOL, previous_tag] += 1
    return tag_counter, word_tag_counter, tag_previous_tag_counter


def bigram_transition_probabilities(tag_previous_tag_counts, tag_counts):
    probabilities = defaultdict(Decimal)
    for tag, previous_tag in tag_previous_tag_counts:
        tag_prev_tag_count = tag_previous_tag_counts[tag, previous_tag]
        prev_tag_count = tag_counts[previous_tag]
        probabilities[tag, previous_tag] = (
            tag_prev_tag_count / prev_tag_count)
    return probabilities


def bigram_word_given_tag_probabilities(word_tag_counts, tag_counts):
    probabilities = defaultdict(Decimal)
    for word, tag in word_tag_counts:
        word_tag_count = word_tag_counts[word, tag]
        tag_count = tag_counts[tag]
        probabilities[word, tag] = word_tag_count / tag_count
    return probabilities


def logsum(operands):
    a_val = max(operands)
    exponents = [math.exp(x - a_val) for x in operands]
    return a_val + math.log(sum(exponents))


def em_forward_backward(observations, tags, trans_prob, emission_prob):
    converged = False
    while not converged:
        # E-step
        gamma = defaultdict(Decimal)
        xi = defaultdict(Decimal)
        delta = Decimal(0.0)
        for index, sentence in enumerate(observations):
            alpha, beta, p_val = forward_backward(sentence, tags, trans_prob, emission_prob)
            print ("\t\t Sentence %d,  p = %.6g"%(index, p_val))
            sentence_length = len(sentence)
            for time_step, word in enumerate(sentence):
                for tag in tags:
                    gamma[word, tag] += alpha[time_step, tag] * beta[time_step, tag] / p_val
                    #if tag != END_SYMBOL:
                    if time_step != sentence_length-1:
                        for prev_tag in tags:
                            xi[tag, prev_tag] += alpha[time_step, prev_tag] * trans_prob.get((tag, prev_tag), LOW_PROB) * emission_prob.get((sentence[time_step+1], tag), LOW_PROB) * beta[time_step+1, tag] / alpha[sentence_length, END_SYMBOL]
        # M-step
        for tag in tags:
            # a-hat
            for prev_tag in tags:
                # Updating transmission probabilities
                trans_prob[tag, prev_tag] = xi[tag, prev_tag] / sum([xi[tag_prime, prev_tag] for tag_prime in tags])
            # b-hat
            for sentence in observations:
                for word in sentence:
                    old_emission = emission_prob[word, tag]
                    emission_prob[word, tag] = gamma[word, tag] / sum([gamma[word_prime, tag] for (word_prime, tag_prime) in gamma if tag_prime == tag])
                    delta += abs(old_emission - emission_prob[word, tag])
        print ("Finished, Delta: %.10g"%(delta))
        converged = delta < DELTA_LIMIT

    return trans_prob, emission_prob
        

def forward_backward(observation, tags, trans_prob, emission_prob):
    observation_length = len(observation)

    # Forward part
    forward_matrix = {}
    first_word = observation[0]
    for tag in tags:
        forward_matrix[0, tag] = (
            trans_prob.get((tag, START_SYMBOL), LOW_PROB) * emission_prob.get((first_word, tag), LOW_PROB))
    # Examine the second word now, we already examined first word
    for time_step, word in enumerate(observation[1:], start=1):
        for tag in tags:
            operands = [forward_matrix[time_step-1, prev_tag] *
                        trans_prob.get((tag, prev_tag), LOW_PROB) * emission_prob.get((word, tag), LOW_PROB)
                        for prev_tag in tags]
            forward_matrix[time_step, tag] = sum(operands)

    operands = [forward_matrix[observation_length-1, tag] *
                trans_prob.get((END_SYMBOL, tag), LOW_PROB) for tag in tags]
    forward_matrix[observation_length, END_SYMBOL] = sum(operands)


    # Backward part
    backward_matrix = {}
    # Initialize
    for tag in tags:
        backward_matrix[observation_length, tag] = trans_prob.get((END_SYMBOL, tag), LOW_PROB)
    # Recursion
    for time_step in range(observation_length-1, -1, -1):
        for prev_tag in tags:
            next_word = observation[time_step]
            operands = [trans_prob.get((tag, prev_tag), LOW_PROB) *
                        emission_prob.get((next_word, tag), LOW_PROB) *
                        backward_matrix[time_step+1, tag] for tag in tags]
            backward_matrix[time_step, prev_tag] = sum(operands)
    P_val = sum([trans_prob.get((tag, START_SYMBOL), LOW_PROB) * emission_prob.get((observation[0], tag), LOW_PROB) * backward_matrix[1, tag] for tag in tags])
    
    return forward_matrix, backward_matrix, P_val


def get_training_info(training_set):
    tag_counts, word_tag_counts, tag_prev_tag_counts = bigram_counters(training_set)
    p_word_given_tag = bigram_word_given_tag_probabilities(word_tag_counts, tag_counts)
    p_tag_given_prev_tag = bigram_transition_probabilities(tag_prev_tag_counts,
                                                           tag_counts)
    tags = tag_counts.keys()
    return tags, p_tag_given_prev_tag, p_word_given_tag


def strip_tags_from_sentences(tagged_sentences):
    sentences = []
    for tagged_sentence in tagged_sentences:
        sentence = [word for word, tag in tagged_sentence]
        sentences.append(sentence)
    return sentences

def bigram_viterbi(observation, tags, transition_prob, emission_prob):
    viterbi_matrix = {}
    backpointer = {}
    observation_length = len(observation)
    for tag in tags:
        first_word = observation[0]
        viterbi_matrix[0, tag] = (
        transition_prob.get((tag, START_SYMBOL), LOW_PROB) + emission_prob.get(
            (first_word, tag), LOW_PROB))
        backpointer[0, tag] = None
    # Examine the second word now, we already examined first word
    for time_step, word in enumerate(observation[1:], start=1):
        for tag in tags:
            # max probability for the viterbi matrix
            viterbi_matrix[time_step, tag] = max(
                viterbi_matrix[time_step - 1, prev_tag] +
                transition_prob.get((tag, prev_tag), LOW_PROB) +
                emission_prob.get((word, tag), LOW_PROB) for prev_tag in tags)
            # argmax for the backpointer
            probability, state = max(
                (viterbi_matrix[time_step - 1, prev_tag] +
                 transition_prob.get((tag, prev_tag), LOW_PROB),
                 prev_tag) for prev_tag in tags)
            backpointer[time_step, tag] = state
    # Termination steps
    viterbi_matrix[observation_length, END_SYMBOL] = max(
        viterbi_matrix[observation_length - 1, tag] +
        transition_prob.get((END_SYMBOL, tag), LOW_PROB) for tag in tags)
    probability, state = max(
        (viterbi_matrix[observation_length - 1, tag] +
         transition_prob.get((END_SYMBOL, tag), LOW_PROB), tag) for tag in tags)
    backpointer[observation_length, END_SYMBOL] = state

    # Return backtrace path...
    backtrace_path = []
    previous_state = END_SYMBOL
    for index in range(observation_length, 0, -1):
        state = backpointer[index, previous_state]
        backtrace_path.append(state)
        previous_state = state
    # We are tracing back through the pointers, so the path is in reverse
    backtrace_path = list(reversed(backtrace_path))
    return backtrace_path


def main():
    # Just so my virtualenv knows where to look for the corpus
    nltk.data.path.append('/home/hill1303/Documents/cse5525/nltk_data')
    full_data = nltk.corpus.treebank.tagged_sents()
    full_training_set = full_data[0:3500]
    training_set1 = full_training_set[0:10]
    training_set2 = full_training_set[1750:]
    test_set = full_data[3500:]

    print('Geting counts')
    tags, tag_given_prev_tag, word_given_tag = get_training_info(training_set1)
    observations = strip_tags_from_sentences(training_set1)

    print('Starting EM forward-backward')
    trans_prob, emission_prob = em_forward_backward(observations, tags, tag_given_prev_tag, word_given_tag)
    print('Finished EM forward-backward')

def test_model(test_set,tags, tag_emission_prob):
    num_error = 0
    total_words = 0
    labeled_test_set = []
    for test_instance in test_set:
        observation = []
        for (word, tag) in test_instance:
            observation.append(word)

        path = bigram_viterbi(observation, tags, trans_prob,
                       emission_prob)

        i = 0
        viterbi_labeled_sent = []
        for (word, tag) in test_instance:
            viterbi_labeled_sent.append((word, path[i]))
            if tag != path[i]:
                num_error += 1
            i = i + 1
        total_words += i
        labeled_test_set.append(viterbi_labeled_sent)
    print (str('Test: ') + " Error Rate: " + str(
        float(num_error) / total_words) + "\n")

if __name__ == '__main__':
    main()
