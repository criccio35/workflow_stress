* Written by Camila Riccio
* Last update: 04/08/21 

Condition-specific co-expression network analysis
-------------------------------------------------

This is a Python implementation of a workflow to analyze co-expression networks under a specific condition. The workflow takes as input the RNA sequencing read counts and phenotypical data of different genotypes, measured under control and stress conditions. It outputs a reduced group of genes marked as relevant for stress response. The workflow aims to identify specific modules
of overlapping communities underlying the co-expression network of genes. Module detection is achieved by using Hierarchical Link Clustering (HLC). LASSO regression is employed as a feature selection mechanism to identify groups of genes that jointly respond to a stress condition.

The structure of the workflow is depicted in file *workflow_structure*.


Setup
------
clone the repository::

  git clone git@github.com:criccio35/workflow_stress.git


Example
-------

The folder **test** contains two examples that illustrate how to use the code to analyze co-expression networks under a stress condition. The file *test_case_study.py* illustrates how the workflow can be used to identify potential saline stress responsive genes in rice (*Oriza sativa*).
The RNA-seq data is available on the GEO database (GSE98455)  `<https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE98455>`, and the phenotypic data is a subset of the supplementary file 1 included in `<https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1006823>`. However the cleaned data can be found in folder **data** inside **test**. Meanwhile file *test_small.py* illustrates how to apply the workflow to a 90% smaller and random RNA-seq dataset for faster and more practical code execution. The outputs of each stage of the workflow (see file *workflow_structure*) are automatically saved in folder **output**.


How to run the example?
^^^^^^^^^^^^^^^^^^^^^^^

To run the example execute the following commands::

  cd test/
  python3 test_small.py
