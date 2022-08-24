# ProjectApricot
The aim of this project is to build an online tool that helps users effortlessly find the important, comprehensive and distilled information about politicians. 

We encode parliament voting records of each Australian senator as a vector in a high-dimensional space. 
We then project each vector onto a two-dimensional plane using dimensionality reduction. The two axis kept after this process closely match the economic and social political spectrum.
The end result is a scatterplot gives an clear visualisation the distribution of political views of Australian senators.

We further investigat the prospects of predicting each senator's 'honesty' by comparing their political view as perceived by the public vs as as indicated by parliament voting records.
To do this we finetuning Google's T5 model to generate most likely political view on a range of policies using on public interview and speech records. The output vectors are processed like mentioned above and the end result is compared with the data generated using official parliament voting records.
In the end, we obtain two vectors for each senator - one encodes the senator's political view in the public's perception, the other encodes how the senator actually votes in parliament. A large difference between the two would indicate that the senator in question votes rather differently then they would have the public belive. 

Unfortunately, our compute power proved insufficient for T5. We could not achieve convergence on even the samllest T5 variant. It was a similar story for GPT-2 and BERT. We also experimented with other, smaller models. However, the results are not good enough. 
We can't afford more computing power :( so unless there is significant improve in the efficiency of these large language models, it would see to be a hard bottleneck.

Therefore, the project is put on hold for now.
