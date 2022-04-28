import numpy as np

class GenericAlgorithm(object):
    def __init__(self, fitness_func, gene_bounds):
        self.gene_length = len(gene_bounds)
        self.gene_bounds = gene_bounds
        self.fitness_func = fitness_func
        self.num_genes = 20
        self.crossover_rate = .5  # likelihood of looking like father
        self.mutation_rate  = .1  # probability of a geno get mutated
        self.rejection_rate = .5

        # auxilary variables
        self.num_rejected = int(self.num_genes * self.rejection_rate)
        self.num_kept = self.num_genes - self.num_rejected

        assert self.rejection_rate > 0 and self.rejection_rate < 1

    def rank_by_fitness(self, genes):
        assert isinstance(genes, np.ndarray) and genes.ndim == 2
        fitness_scores = np.array(list(map(self.fitness_func, genes)))
        sorted_ids = np.argsort(fitness_scores)
        return genes[sorted_ids,:], fitness_scores[sorted_ids]

    def initialize_gene_pool(self):
        gene_seeds = np.random.rand(self.num_genes, self.gene_length)
        for ix, bnd in enumerate(self.gene_bounds):
            if bnd is None:
                continue
            assert bnd[1] > bnd[0]
            gene_seeds[:,ix] = gene_seeds[:,ix] * (bnd[1]-bnd[0]) + bnd[0]
        gene_pool = gene_seeds

        self.new_generation = gene_pool[:self.num_rejected]
        self.old_generation, self.old_generation_fitness_scores = self.rank_by_fitness(gene_pool[self.num_rejected:])

    def selection(self):
        # test new generation
        self.new_generation, self.new_generation_fitness_scores = self.rank_by_fitness(self.new_generation)
        # merge with old generation
        fitness_scores = np.hstack([self.old_generation_fitness_scores, self.new_generation_fitness_scores])
        genes = np.vstack([self.old_generation, self.new_generation])
        # re-rank the merged genes
        sorted_ids = np.argsort(fitness_scores)
        genes, fitness_scores = genes[sorted_ids,:], fitness_scores[sorted_ids]
        # reject bad genes
        return genes[self.num_rejected:,:], fitness_scores[self.num_rejected:]

    def crossover(self, gene_pool):
        new_generation = list()
        for _ in range(self.num_rejected):
            # select a couple from gene_pool
            couple_ids = np.random.choice(range(len(gene_pool)), 2, replace=False)
            couple = gene_pool[couple_ids,:].copy()
            for dim in range(self.gene_length):
                if np.random.rand() < self.crossover_rate:
                    couple[0,dim] = couple[1,dim]
            new_generation.append(couple[0,:])
        return np.vstack(new_generation)

    def mutation(self, gene_pool):
        for gene in gene_pool:
            for dim in range(len(gene)):
                if np.random.rand() < self.mutation_rate:
                    low, high = self.gene_bounds[dim]
                    gene[dim] = np.random.rand() * (high-low) + low
        return gene_pool

    def evolve_one_generation(self, iteration):
        self.old_generation, self.old_generation_fitness_scores = self.selection()
        # display current best gene
        best_gene = self.old_generation[-1,:]
        best_fitness = self.old_generation_fitness_scores[-1]
        # print(self.old_generation)
        # print(self.old_generation_fitness_scores.tolist())
        print("[{}] Best fitness score: {} at {}".format(iteration, best_fitness, best_gene.tolist()))
        # creation of the new generation
        self.new_generation = self.crossover(self.old_generation)
        self.new_generation = self.mutation(self.new_generation)

        return best_gene.copy()

    def evolve(self):
        self.initialize_gene_pool()
        for idx in range(50):
            best_gene = self.evolve_one_generation(idx)
        return best_gene


class DriftAlgorithm(GenericAlgorithm):
    def mutation(self, gene_pool):
        sigma = .1   # according to 69-95-99 rule, 95% probability will fall in the scope of [-2s, +2s] around the center
        for gene in gene_pool:
            for dim in range(len(gene)):
                if np.random.rand() < self.mutation_rate:
                    low, high = self.gene_bounds[dim]
                    center_normalized = (gene[dim]-low) / (high-low)
                    seed = (np.random.randn() * sigma + center_normalized) % 1.  # sample from a circled gaussian distribution, sigma control how
                    gene[dim] = seed * (high-low) + low
        return gene_pool