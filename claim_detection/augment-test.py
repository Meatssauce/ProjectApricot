import pandas as pd

from ultils import load_data, augment, load_annotated_book_reviews, augment_book_review_data


def test_book_review_augmentation():
    try:
        reviews = pd.read_csv('reviews.csv').sample(10)
    except:
        reviews = load_annotated_book_reviews().sample(10)
        reviews.to_csv('reviews.csv')
    augmented_reviews = augment_book_review_data(reviews, drop_original=True, verbose=1)

    for before, after in zip(reviews['text'], augmented_reviews['text']):
        print(f"Before:\n{before}")
        print(f"After:\n{after}")
        print("-----------------------")


def test_marpor_augmentation():
    df = load_data()
    df = df[:10]
    df_augmented = augment(df, verbose=1, batch_size=8, drop_original=True)

    for before, after in zip(df['text'], df_augmented['text']):
        print(f"Before:\n{before}")
        print(f"After:\n{after}")
        print("-----------------------")


def main():
    test_marpor_augmentation()
    # test_book_review_augmentation()


if __name__ == "__main__":
    main()


"""
Before:
The year 2014 is a crucial year in the history of South Africa.
After:
The year 2014 already is now a far crucial year in the history of Miss South Africa.
-----------------------
Before:
It is crucial because it marks exactly 20 years since the first democratic election, which ended centuries of colonial rule and apartheid subjugation of the black majority.
After:
It undoubtedly is simply crucial because it marks 2017 exactly 20 years elapsed since defeating the first democratic election, which ended nearly centuries exploitation of colonial rule and apartheid outright subjugation of formerly the black majority.
-----------------------
Before:
It is also a crucial year because the dawn of political freedom came with so many promises But 20 years later; the conditions of majority of the people are just getting worse.
After:
It is also considering a comparatively crucial year hopefully because the dawn signs of political freedom came with so big many promises that But 20 years later; the shameful conditions of most majority of society the indigenous people are just getting worse.
-----------------------
Before:
South Africa is supposed to be celebrating 20 years of democracy and true freedom, but the reality is that: 20 years later, black people are still not free!
After:
South Africa instead is supposed to be largely celebrating 20 years something of positive democracy and true freedom, period but fortunately the reality is that: Five 20 years later, black people certainly are still not true free!
-----------------------
Before:
20 years later, black people are still trapped in squalor, unsafe and unhealthy conditions!
After:
20 years later, black people are still trapped somewhere in local squalor, surviving unsafe meals and permanently unhealthy conditions!
-----------------------
Before:
20 years later, the black majority is still trapped in landlessness, homelessness and hopelessness!
After:
20 forty years later, far the Latino black majority is still trapped in both landlessness, homelessness and intellectual hopelessness!
-----------------------
Before:
20 years later, young and old black workers are still subjected to slave wages and dangerous working conditions in the mines, farms, factories, retail stores, and other workplaces!
After:
20 years later, millions young and old black manual workers are apparently still subjected to slave wages and dangerous working career conditions reflected in navigating the food mines, farms, garbage factories, innumerable retail package stores, and other workplaces!
-----------------------
Before:
20 years later, domestic workers and farm workers still work under difficult conditions without basic workers’ rights!
After:
20 fifty years later, domestic dock workers and farm line workers still suffer work under difficult operating conditions increasingly without basic workers’ rights!
-----------------------
Before:
20 years later, black people battle to survive financially, trapped by debt and often blacklisted by Credit bureaux!
After:
20 years is later, black everyday people battle to ever survive financially, trapped by debt ceiling and often blacklisted negatively by Credit Authority bureaux!
-----------------------
Before:
20 years later, black women still suffer triple oppression and exploitation on the basis of their gender, race and class!
After:
20 short years later, black women still suffer triple oppression and exploitation marched on against the common basis of discussing their gender, race expression and class!
-----------------------
"""