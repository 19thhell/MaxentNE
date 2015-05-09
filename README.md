Name Entity Recognizer
======================

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

The script will read in the language and test set from STDIN, then train the NER and produce statistics base on the language and test set you choose. The whole process takes about 45 minutes.
