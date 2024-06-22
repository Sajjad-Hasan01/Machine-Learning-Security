import pandas


def algo():
    df = pandas.DataFrame({'X1':[2.5, 3.1, 2.8], 'X2':[None, 12, None]})
    print("\nDF: ", df)

    df1 = df.fillna(df.mean())
    print("\nDF1", df1)

    df2 = df.dropna(axis="rows", inplace= False)
    print("\nDF2: ", df2)

    df3 = df.fillna(5)
    print("\nDF3: ", df3)

if __name__ == '__main__':
    algo()
