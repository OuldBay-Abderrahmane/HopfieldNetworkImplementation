import experiment as exp

sizes = [10, 18, 34, 63, 116, 215, 397, 733, 1354, 2500]


exp.capacity (sizes, "hebbian", 10)
exp.robustness(sizes, "hebbian", 0.05, 2, .0)
exp.capacity (sizes, "storkey", 10)
exp.robustness(sizes, "storkey", 0.05, 2, .0)
#exp.image_retrieval()
