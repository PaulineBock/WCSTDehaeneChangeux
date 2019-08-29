# WCSTDehaeneChangeux
Code for an implementation of the Dehaene and Changeux's model of Wisconsin Card Sorting Test (1991).

Model adaptated from Dehaene and Changeux model (1) , using neural networks, to perform Wisconsin Card Sorting Test and modellize flexibility of decision making. (DehaeneChangeux.py)
Other versions of the model are partially implemented, using the Task Sets concept and for which the representation and functioning is inspired by a thesis of Flora Bouchacourt (2).
There is one with AN and TN learning (TSDehaene.py) and one with AN influenced by perfect TN already learned. (PerfectTNDehane.py)
References
(1) Dehaene, S., & Changeux, J. P. (1991). The Wisconsin Card Sorting Test: Theoretical analysis and modeling in a neuronal network. Cerebral cortex, 1(1), 62-79.
(2) Bouchacourt, F. (2016). Hebbian mechanisms and temporal contiguity for unsupervised task-set learning (Doctoral dissertation).

How to use: 
python DehaeneChangeux.py
python PerfectTNDehaene.py

Numpy and Matplotlib libraries required.
