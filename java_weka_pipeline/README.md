# Java Pipeline

This part of the pipeline is the one running the experiments over the dumped representation of the embeddings via propositionalization for RipperK.

The Java solution is Maven-based. The main class associated to the project will take at least two arguments, where the first is the absolute path where the results were dumped, and the second argoment is the specific dataset over which the user wants to run the tests to (e.g., this matches with the initial name, or several initial names, of the folders within the dump folder). The third and optional argument deals  with parallelism but, due to some requirement over temporary files in Weka, this solution is not advised
