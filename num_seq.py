from __future__ import print_function
import random
import math
import sys
from argparse import ArgumentParser
from operator import itemgetter
from bisect import bisect_right

from pprint import pprint

# problem slightly modified from: http://www.ai-junkie.com/ga/intro/gat3.html
# Given the digits 0 through 9 and the operators +, -, * and /,  find a mathematical
# expression obeying BEDMAS rules and integer division that evaluates to a target number.
# So, given the target number 15, the sequence 6+2*12/2-3 would be one possible solution.

# Genetic learning algorithm (taken from http://www.ai-junkie.com/ga/intro/gat2.html)
#  1. Encode the features of the problem into a set of genes (likely binary)
#  1. Generate a population of chromosomes that combine random genes
#  2. How good is each chromosome at solving the problem? (assign a "fitness score")
#  3. Select 2 chromosomes from the population (selection change proportional to "fitness score")
#  4. Perform gene crossover after random location according to "crossover rate"
#  5. Flip chromosome bits using the "mutation rate"
#  6. Repeat steps 2 - 5 until a newly desired population size

# TODO: The following 4 constants may be overrided by the script call parameters
# (I know it's bad style but it's a stand-in before I convert this script to use classes)
MUTATION_RATE   = 0.03
CROSSOVER_RATE  = 0.7
CHROMO_LEN      = 10     # of genes
POP_SIZE        = 100

OPERATORS       = ['+', '-', '*', '/']
NUMBERS         = range(10)
GENES           = OPERATORS + NUMBERS + [None]   # all possible genes

GENE_BIT_MAPPING = { gene: i for i, gene in enumerate(GENES) }
BIT_GENE_MAPPING = { i: gene for gene, i in GENE_BIT_MAPPING.iteritems() }

# number of bits used to represent each gene
# need to know this when we perform mutation so
# we generate a bit mask with the right number of bits
GENE_BIT_COUNT = int(math.floor(math.log(len(GENES), 2)) + 1)

def chromo_to_expr(chromo):
  """chromo: a list of ints that can be mapped to genes"""
  return ''.join([
    str(BIT_GENE_MAPPING[g])
    for g in chromo
    if BIT_GENE_MAPPING[g] is not None
  ])

def eval_expression(expr):
  """
  expr is a sequence of genes defined by GENES.  Return the evaluation of
  the given mathematical expression. None genes are ignored in the chromosome
  but everything else is concatenated into a string and then evaluated using
  python 'eval' method.  Errors will return None, otherwise the result is returned.
  """
  # remove None values and then concatenate them all as a string
  try:
    return eval(expr)
  except (ZeroDivisionError, SyntaxError):
    return None

def fitness_score(chromo, target_sum):
  chromo_eval = eval_expression(chromo_to_expr(chromo))

  # useless chroosone
  if chromo_eval == None:
    return 0

  # exact match
  if chromo_eval == target_sum:
    return float('inf')

  try:
    return 1. / abs(target_sum - chromo_eval)
  except OverflowError:
    return 0 # target_sum - chromo_eval can be extremely large sometimes

def create_chromosome():
  chromo = []
  for i in xrange(CHROMO_LEN):
    chromo.append(random.choice(GENE_BIT_MAPPING.values()))
  return chromo

def crossover(chromo_a, chromo_b):
  """
  do crossover at gene-level, not bit-level within genes.
  both chromosomes are assumed to be of the same length.
  Returns (new_chromo_a, new_chromo_b) which had crossover performed.
  """
  # choose random crossover point
  cross_point = int(round(random.random()) * len(chromo_a))
  new_chromo_a = chromo_a[:cross_point] + chromo_b[cross_point:]
  new_chromo_b = chromo_b[:cross_point] + chromo_a[cross_point:]
  return (new_chromo_a, new_chromo_b)

def mutate(chromo, mutation_rate):
  """Returns a new mutated chromosome from the given chromosome"""
  mutated_chromo = []
  for gene in chromo:
    while True:
      # create the xor bit mask for performing mutation like '010001' where 1
      # means that a bit in the gene will be mutated (flipped)
      mutation_bits = sum(
          2**x
          for x in xrange(GENE_BIT_COUNT)
          if random.random() >= mutation_rate
      )
      mutated_gene = gene ^ mutation_bits

      # the mutation must produce a valid gene
      if mutated_gene in BIT_GENE_MAPPING.iterkeys():
        break

    mutated_chromo.append(mutated_gene)
  return mutated_chromo

def breed_generation(chromo_scores):
  """
  Breed a new population where the chromosomes with the higher fitness
  scores are more likely to be chosen for reproduction
  (where crossover and mutation may occur).  Roulette wheel selection is used
  to perform selection.  The new population size will be the same as the old,
  rounded down to the nearest even amount.

  chromo_scores: list of (chromo, score) tuples where the score must be +ve
  """
  next_gen_chromos = []
  min_score = min(c[1] for c in chromo_scores)
  max_score = max(c[1] for c in chromo_scores)
  total_score = sum(c[1] for c in chromo_scores)

  # normalize and distribute scores between [0, 1)
  # the bigger the interval between the prev entry and
  # current entry means the bigger chance to reproduce
  distributed_chromo_scores = []
  cumulative_norm_score = 0

  for chromo_tuple in chromo_scores:
    # if everything had a score of 0, give everything an even chance
    if total_score != 0:
      norm_score = float(chromo_tuple[1]) / total_score
    else:
      norm_score = 1. / len(chromo_scores)

    distributed_chromo_scores.append(norm_score + cumulative_norm_score)
    cumulative_norm_score += norm_score

  def roulette_chromo():
    r = random.random()
    chromo_index = bisect_right(distributed_chromo_scores, r)

    # there is a small chance due to floating point rounding when creating
    # the distributed_chromo_scores that the last one might not be 1.00 so
    # random() might be higher than it.  Print warning as this might also
    # be a sign of a bug in making the chromo scores.
    if chromo_index >= len(chromo_scores):
      print('WARNING, tried to select chromosome with relative '
          + 'fitness score <= %s but the highest is %s, using the highest one anyways...' %
            (r, distributed_chromo_scores[-1]), file=sys.stderr)
    return chromo_scores[min(chromo_index, len(chromo_scores) - 1)][0]

  # crossover and mutate two chromosomes from the roulette wheel
  for i in xrange(len(chromo_scores) / 2):
    chromo_a, chromo_b = crossover(roulette_chromo(), roulette_chromo())
    next_gen_chromos.append(mutate(chromo_a, MUTATION_RATE))
    next_gen_chromos.append(mutate(chromo_b, MUTATION_RATE))

  return next_gen_chromos


def main():
  # TODO: I know, bad practice to do this but it's a stand in before I convert this
  # script to classes
  global MUTATION_RATE, CROSSOVER_RATE, POP_SIZE, CHROMO_LEN

  parser = ArgumentParser(description=
    'Given the digits 0 through 9 and the operators +, -, * and /,  find a mathematical ' +
    'expression obeying BEDMAS rules and integer division that evaluates to a target number.' +
    'So, given the target number 15, the sequence 6+2*12/2-3 would be one possible solution.'
  )
  parser.add_argument('x', type=int, help='target expression value')
  parser.add_argument('-m', '--mutation-rate', type=float,
    help='chance of gene bits mutating after reproduction',
    default=MUTATION_RATE)
  parser.add_argument('-c', '--crossover-rate', type=float,
    help='chance of performing crossover between two reproducing chromosomes',
    default=CROSSOVER_RATE)
  parser.add_argument('-p', '--pop-size', type=int,
    help='number of chromosomes in each generation',
    default=POP_SIZE)
  parser.add_argument('-l', '--chromo-len', type=int,
    help='number of genes in a chromosome',
    default=CHROMO_LEN)

  args = parser.parse_args()

  target_val      = args.x
  MUTATION_RATE   = args.mutation_rate
  CROSSOVER_RATE  = args.crossover_rate
  POP_SIZE        = args.pop_size
  CHROMO_LEN      = args.chromo_len

  chromos = [ create_chromosome() for i in xrange(POP_SIZE) ]
  generation = 1

  while True:
    chromo_scores = [
      (chromo, fitness_score(chromo, target_val))
      for chromo in chromos
    ]

    print('Gen {0:<5}, high score {1:<8}'.format(
      generation, max(c[1] for c in chromo_scores)))

    # print a couple chromosomes every couple generations to see what they're like
    if generation % 100 == 0:
      pprint([chromo_to_expr(chromo) for chromo in chromos[:5]])

    perfect_chromos = [
      chromo
      for chromo, score
      in chromo_scores if score == float('inf')
    ]

    if len(perfect_chromos) >= 1:
      break

    chromos = breed_generation(chromo_scores)
    generation += 1

  print('Found %s perfect expressions for "%s" in %s generations of size %s' %
      (len(perfect_chromos), target_val, generation, POP_SIZE))
  print('Solutions:')
  print('\n'.join([chromo_to_expr(chromo) for chromo in perfect_chromos]))

if __name__ == '__main__':
  main()
