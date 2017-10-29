import numpy as np
import pandas as pd
import os


def main():
    df = pd.read_csv('Tools/Titanic.csv', index_col='PassengerId')

    # print df['Sex'].value_counts()
    # print df.Survived.head()
    # print df.sort_values(['Survived'], ascending=True).head(1)
    # print df.Survived.size
    # print np.sum(df.Survived)
    # df.isnull().any().any()

    passengers_count = df.shape[0]
    # First question
    male_count = df['Sex'][df['Sex'] == 'male'].size
    female_count = df['Sex'][df['Sex'] == 'female'].size
    create_answers('1', '%s %s' % (male_count, female_count))
    # Second question
    survived_proportion = float(np.sum(df.Survived)) / float(df.Survived.size) * 100
    create_answers('2', '%.2f' % np.round(survived_proportion, 2))
    # Third question
    first_class_count = df.Pclass[df.Pclass == 1].size
    first_class_proportion = float(first_class_count) / float(passengers_count) * 100
    create_answers('3', '%.2f' % np.round(first_class_proportion, 2))
    # Fourth question
    age_mean = np.mean(df.Age)
    age_median = np.nanmedian(df.Age)
    create_answers('4', '%.2f %.2f' % (age_mean, age_median))
    # Fifth question
    pearson_corr = df.SibSp.corr(df.Parch, method='pearson')
    create_answers('5', '%.2f' % round(pearson_corr, 2))
    # Sixth question
    df['FirstName'] = df.Name.str.extract('(Mr\. |Miss\. |Master. |Mrs\.[A-Za-z ]*\()([A-Za-z]*)')[1]
    female_frequently_name = df.FirstName[df.Sex == 'female'].mode()[0]
    create_answers('6', '%s' % female_frequently_name)


def create_answers(answer_number, answer):
    answers_path = 'Answers'
    if not os.path.exists(answers_path):
        os.makedirs(answers_path)
    with open('Answers/%s.txt' % answer_number, 'w+') as f:
        f.write(answer)


if __name__ == '__main__':
    main()
