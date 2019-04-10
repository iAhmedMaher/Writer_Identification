# Writer_Identification

This repository is for the task of identifying a writer of unknown form given forms to train on for all the possible writers.
The technique implemented yields accuracy of 98% on IAM given the following settings:
We have 3 writers, each with 2 forms to train on. Then, we have 1 form which we want to know which writer for the 3 wrote it. The model can't be trained before hand; it must only be trained on those 3 writers' training forms.
