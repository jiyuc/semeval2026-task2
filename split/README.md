## Data statistics for the Train-Test Split
- subtask1
  - Fold 1
    - Train users: 109; Test users: 28
    - Train: 2357; Test: 407
  - Fold 2
    - Train users: 109; Test users: 28
    - Train: 2004; Test: 760
  - Fold 3
    - Train users: 110; Test users: 27
    - Train: 2208; Test: 556
  - Fold 4
    - Train users: 110; Test users: 27
    - Train: 2282; Test: 482
  - Fold 5
    - Train users: 110; Test users: 27
    - Train: 2205; Test: 559
- subtask2a
  - Fold 1
    - Train users: 109; Test users: 28
    - Train: 2165; Test: 599
  - Fold 2
    - Train users: 109; Test users: 28
    - Train: 2320; Test: 444
  - Fold 3
    - Train users: 110; Test users: 27
    - Train: 2187; Test: 577
  - Fold 4
    - Train users: 110; Test users: 27
    - Train: 2290; Test: 474
  - Fold 5
    - Train users: 110; Test users: 27
    - Train: 2094; Test: 670
- subtask2b
  - Fold 1
    - Train users: 109; Test users: 28
    - Train: 2165; Test: 599
  - Fold 2
    - Train users: 109; Test users: 28
    - Train: 2320; Test: 444
  - Fold 3
    - Train users: 110; Test users: 27
    - Train: 2187; Test: 577
  - Fold 4
    - Train users: 110; Test users: 27
    - Train: 2290; Test: 474
  - Fold 5
    - Train users: 110; Test users: 27
    - Train: 2094; Test: 670
- subtask2b_detailed
  - Fold 1
    - Train users: 109; Test users: 28
    - Train: 2165; Test: 599
  - Fold 2
    - Train users: 109; Test users: 28
    - Train: 2320; Test: 444
  - Fold 3
    - Train users: 110; Test users: 27
    - Train: 2187; Test: 577
  - Fold 4
    - Train users: 110; Test users: 27
    - Train: 2290; Test: 474
  - Fold 5
    - Train users: 110; Test users: 27
    - Train: 2094; Test: 670
- subtask2b_detailed
  same as subtask2b



## Data Format
### Subtask 1 — Longitudinal Affect Assessment:

The training data (subtask1_[train|test].csv) has the following format:

| user_id         | text_id           | text                                   | timestamp          | collection_phase             | is_words           | valence                | arousal      |
|-----------------|-------------------|----------------------------------------|--------------------|------------------------------|--------------------|------------------------|-------------|
| example_user_id | example_text_id    | example text                           | example timestamp  | [1, 2, 3, 4, 5, 6, 7]        | [True, False]      | [-2, 1, 0, 1, 2]       | [0, 1, 2]   |
where,
- user_id: anonymous identifier for the author of the texts (essays/feeling words)
- text_id: identifier for a specific text written by an author
- text: the essay or the feeling words written by an author
- timestamp: when the text was written
- collection_phase: data collection phase (1–7)
- is_words: boolean; False for essays, True for feeling words
- valence: valence score associated with the text
- arousal: arousal score associated with the text


### Subtask 2A - Forcasting State Change 

| user_id |state_change_valence|state_change_arousal|
|---------|--------------------|--------------------|
| e.g., 1 | e.g., 0.42	 |e.g., -0.17|

where,
- `state_change_valence` is computed per user by subtracting the valence of the current text from the valence of the following text. Texts are sorted in ascending temporal order per user using timestamp. For each user, the value is NaN for their last text.
- `state_change_arousal` is computed per user by subtracting the arousal of the current text from the arousal of the following text. Texts are sorted in ascending temporal order per user using timestamp. For each user, the value is NaN for their last text.


### Subtask 2B - Forcasting Disposition Change
#### Definition of Dispositional Affect [[ref. Wikipedia](https://en.wikipedia.org/wiki/Dispositional_affect)]

Dispositional affect, similar to mood, is a personality trait or overall tendency to respond to situations in stable, predictable ways. This trait is expressed by the tendency to see things in a positive or negative way. People with high positive affectivity tend to perceive things through "pink lens" while people with high negative affectivity tend to perceive things through "black lens". The level of dispositional affect affects the sensations and behavior immediately and most of the time in unconscious ways, and **its effect can be prolonged (between a few weeks to a few months)**.

|group|disposition_change_valence|disposition_change_arousal|
|--------|--------------------|--------------------|
|e.g., 2	|e.g., -0.31	|e.g., 0.08
where,
- group is the marker to designate texts per user into two halves, with group=1 being the first half for a user and group=2 being the second half for that user.
- disposition_change_valence is computed per user by subtracting the mean valence of the first half of their texts (maked as group 1) from the mean valence of the second half of their texts. Texts are sorted in ascending temporal order per user using timestamp.
- disposition_change_arousal is computed per user by subtracting the mean arousal of the first half of their texts from the mean arousal of the second half of their texts (maked as group 2). Texts are sorted in ascending temporal order per user using timestamp.

### Two additional files for subtask2b:
> Note 2: We also provide additional columns in the detailed file train_subtask2b_detailed.csv to show the intermediate values used for computing the released labels: text_num, num_texts_per_user, group, mean_valence_half1, mean_valence_half2, mean_arousal_half1, mean_arousal_half2.

> Note 3: We also provide a trimmed version with only user_id and disposition_change columns (train_subtask2b_user_disposition.csv): user_id, disposition_change_valence, disposition_change_arousal. 
