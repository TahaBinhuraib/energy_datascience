def create_inout_sequences(input_data, timeStamps):
    inout_seq = []
    L = len(input_data)
    for i in range(L - timeStamps):
        train_seq = input_data[i : i + timeStamps]
        train_label = input_data[i + timeStamps : i + timeStamps + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq
