# Citation
The data used here is from the corpus dataset of the [Manifesto Project](https://manifesto-project.wzb.eu/).

## Instructions
- Structure: {'text', 'cmp_code', 'eu_code'}, where
  - 'text' is a quasi-sentence (see quick primer on what constitutes a quasi-sentence)
  - 'cmp_code' represents the corresponding political topic, a.k.a. the 'label'
  - 'eu_code' is not needed for our case and should be removed
- headlines are labelled as 'H'
- examples that don't express a political opinion are labelled as NAN
- examples that expresses a political opinion that does not belong to an existing category are labelled as '000'
- The dataset uses UTF-8 encoding
- You need to place csv files downloaded from Manifesto Project website in appropriately named folders based on country and time period (e.g. AU 2001-2021) for load_data() to work properly

Click [here](https://manifesto-project.wzb.eu/tutorials/primer) for a quick primer on reading and using the dataset.
