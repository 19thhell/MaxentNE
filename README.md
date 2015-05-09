Name Entity Recognizer
======================

Objective
---------
Build a name entity recognizer to extract name entities in CoNLL-2002 dataset using maximum entropy classifier.

Result
------
The program produce a NER that has F1-score 73.60% on Spanish corpus and 69.26% on Dutch corpus, while previous best result using similar technique has F1-score 73.66% on Spanish corpus and 68.08% on Dutch corpus.

Prerequisites
-------------
Python 2.6, NLTK 3.0.2, Megam.

File
----
**MaxentNE.pdf**: Project write-up.

**maxent.py**: Main code for producing NER.

**megam_i686.opt**: Megam executable.

How to Use
----------
In CIMS server terminal, run the following command:

```python maxent.py```

The script will read in the choice of language and test set from STDIN, then train the NER and produce statistics base on the language and test set you choose. The whole process takes about 45 minutes.
