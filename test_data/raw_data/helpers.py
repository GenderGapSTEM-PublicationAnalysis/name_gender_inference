def clean_name_part(df, name_part="middle_name"):
    """keep the string in column 'middle_name' if it has more than one character.
    Otherwise replace by ''. """

    def try_to_simplify(s):
        try:
            if len(s) > 1:
                return s.lower()
            else:
                return ''
        except:
            return ''

    df[name_part] = df[name_part].map(lambda x: try_to_simplify(x))


def build_full_name(df):
    df["full_name"] = df.apply(lambda x: x.first_name + ' ' + x.middle_name + ' ' + x.last_name, axis=1)
    df.full_name = df.full_name.str.replace('  ', ' ')  # if no middle_name then the above line yields 2 empty spaces
