# slot-tagging
HW1 Report

Yifei Gan

1 Introduction

The task is to develop a Named Entity Recognition (NER) model for slot tagging in a virtual personal assistant’s natural language utterances. The goal is to identify specific entities (slots) such as a director’s name or a movie’s release year from user requests to fulfill backend queries. For example, given a user request like, “Show me movies directed by Woody Allen recently,” the model should tag "Woody Allen" as a director and "recently" as a release\_year. These slots help the virtual assistant parse the user’s intent and retrieve the necessary information to generate an appropriate response.

This is a supervised sequence labeling problem, where the goal is to predict labels (slots) for each token in a given input sentence. The model is trained on labeled data, where each token in the training utterances is annotated with its corresponding slot tag, using the IOB (Inside- Outside-Beginning) tagging format. The task is evaluated on its accuracy in assigning correct slot labels to tokens in unseen sentences.

The model must perform token-level classification to assign one of the following labels to each token in an utterance:

O (Outside): No slot associated with this token. B\_<slot\_type> (Beginning): The first token of a slot entity (e.g., B\_director for the first token of a director’s name). I\_<slot\_type> (Inside): Any additional tokens within a slot entity (e.g., I\_director for subsequent tokens in a director’s name). This slot tagging task is a crucial compo-

ID utterances![](Aspose.Words.6d7ba103-f8fa-44ea-bed5-26f894df78ec.001.png)

0 star of thor

1  who is in the movie the campaign
1  list the cast of the movie the campaign ![](Aspose.Words.6d7ba103-f8fa-44ea-bed5-26f894df78ec.002.png)Table 1: Example of testing dataset.

nent of Natural Language Understanding (NLU) in dialogue systems, enabling the virtual assistant to extract structured information from unstructured user input.

2 Model

The training data is a set of user utterances labeled with IOB-formatted slot tags. Each utterance is tokenized, and the slot tags are assigned to the cor- responding tokens. Labels and tokens are encoded into integer representations using Scikit-Learn’s LabelEncoder. I design an LSTM-based sequence labeling model with an embedding layer, a bidi- rectional LSTM layer, and a fully connected layer. Embedding Layer converts each word into a dense vector representation to capture semantic similarity. LSTMLayerisabidirectionalLSTMcapturesboth forward and backward context in the utterance, and Fully Connected Layer that maps the LSTM output to the number of slot classes, producing tag scores for each word.

For training procedure, the model is trained using a cross-entropy loss function, optimized with Adam. For each epoch, the model iterates over the sen- tences in the training set, computing the loss and updating parameters through backpropagation. Af- ter each epoch, the average loss and token-level accuracy are recorded. During evaluation, loss and accuracy are plotted over the epochs to monitor model performance. Accuracy is calculated as the proportion of correctly predicted slot tags for to- kens in each batch.

For testing, an independent test set is provided. A predict\_tags function encodes test utterances, for- wards them through the model, and decodes the resulting tags into IOB format. Each word in the test set is assigned a slot tag, and the predictions are formatted as required for submission. The final predictions are saved to a CSV file, with each row

ID utterances IOB Slot tags![](Aspose.Words.6d7ba103-f8fa-44ea-bed5-26f894df78ec.003.png)

0 who plays luke on star wars new hope O O B\_char O B\_movie I\_movie I\_movie I\_movie 1 show credits for the godfather O O O B\_movie I\_movie

2 who was the main actor in the exorcist O O O O O O B\_movie I\_movie![](Aspose.Words.6d7ba103-f8fa-44ea-bed5-26f894df78ec.004.png)

Table 2: Example of training dataset.

containing an utterance ID, the original utterance, and the predicted IOB slot tags. The output file serves as a submission for evaluation.

3 Experiments

To locate which measure of hyperparameters is the most accurate, optuna has been set and used in order to help.

3\.1 Hyperparameter Tuning

Before using optuna to find the best parameters, the parameters I use are:

{embedding\_dim: 128,

hidden\_dim: 256,

learning\_rate: 0.001,

epochs: 5}

For optuna, I choose the range of parameters as following:

![](Aspose.Words.6d7ba103-f8fa-44ea-bed5-26f894df78ec.005.png)

Figure 1: Accuracy and Loss for original model

![](Aspose.Words.6d7ba103-f8fa-44ea-bed5-26f894df78ec.006.png)

embedding\_dim = trial.suggest\_int( ' embedding\_dim' Figure 2: Accuracy and Loss for refined model

, 64, 256)

hidden\_dim = trial.suggest\_int( ' hidden\_dim ' , 128, 512)

learning\_rate = trial.suggest\_loguniform( ' learning\_rate ' 5 Reference

, 1e-5, 1e-2)

epochs = trial.suggest\_int( ' epochs ' , 5, 20)

Aggarwal, T. (n.d.). Mastering named entity recognition: Unveiling techniques for accu-

In total, there’s 49 finished trails and the best pa- rate entity extraction in NLP. Tech Ladder. rameters set is: https://www.techladder.in/article/mastering-

{embedding\_dim: 160, named-entity-recognition-unveiling-techniques- hidden\_dim: 157, accurate-entity-extraction-nlp

learning\_rate: 0.000332992446379797, Kuriakose, J. (2019, December 25). Bio epochs: 19} tagged text to original text. Medium. All models were evaluated based on accuracy and https://medium.com/analytics-vidhya/bio-tagged- lost during training and validation. text-to-original-text-99b05da6664

4 Results

The testing accuracy for original model is 0.72425, the training accuracy is 0.9954, loss is 0.0190. The testing accuracy for refined model is 0.75941, and the training accuracy is 0.9997, loss is 0.0014.
