from lm_util import read_from_file, get_char_dicts, UNK_ID
from CharLM import LSTM
import argparse
import dynet as dy
import random
import math


def create_parser():
    path = "/Users/thomaszenkel/GitRepos/MasterThesis/bin/CTC/"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="train lstm"
    )

    parser.add_argument(
        "--name", default="test2",
        help="model will be saved under name.model"
    )

    parser.add_argument(
        "--load_model", default="",
        help="model to load"
    )

    parser.add_argument(
        '--batch_size', type=int, default=32,
        help="batch_size"
    )

    parser.add_argument(
        '--max_sentence_length', type=int, default=128,
        help="batch_size"
    )

    parser.add_argument(
        '--layers', type=int, default=1,
        help="number of layers of lstm"
    )

    parser.add_argument(
        '--hiddenUnits', type=int, default=1024,
        help="number of hidden units of lstm"
    )

    parser.add_argument(
        '--dropout', type=int, default=0,
        help="apply dropout or not"
    )

    parser.add_argument(
        '--adam', type=int, default=0,
        help="adam or not"
    )

    parser.add_argument(
        '--train', default=path + "valid.txt"#"trainLMUniq.txt"
    )

    parser.add_argument(
        '--valid', default=path + "valid.txt"
    )

    parser.add_argument(
        '--test', default=path + "test.txt"
    )

    parser.add_argument(
        '--lr', type=float, default=0.1,
        help="learning rate, only used for sgd"
    )

    # dynet fix
    parser.add_argument('--dynet-mem')
    parser.add_argument('--dynet-gpu')

    return parser


def calc_avg_loss(data):
    number_of_batches = int(len(data) / args.batch_size)
    loss_sum, number_of_chars = 0.0, 0.0
    for i in range(number_of_batches):
        batch = data[i * args.batch_size:(i + 1) * args.batch_size]
        number_of_chars += len(batch) * (len(batch[0]) - 1) # start of sentence symbol gets ignored in loss
        loss_sym = lstm.loss_of_sentences(batch)
        loss_sum += loss_sym.value()
    return loss_sum / number_of_chars


def sample(lstm, sentence):
    char_ids = [UNK_ID] + list(map(lambda c: char_dict[c], sentence))
    char_ids, probs = lstm.generate(char_ids)
    print("".join(map(lambda i: int_to_char[i], char_ids)))


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    print(args, flush=True)

    char_dict, int_to_char = get_char_dicts()
    vocab_size = len(int_to_char)

    model = dy.Model()
    if args.load_model != "":
        lstm = model.load(args.load_model)[0]
    else:
        lstm = LSTM(model, len(int_to_char), num_layers=args.layers, hidden_dim=args.hiddenUnits)

    if args.adam == 1:
        trainer = dy.AdamTrainer(model)
    else:
        trainer = dy.SimpleSGDTrainer(model, e0=args.lr)

    test_data = read_from_file(args.test, char_dict, args.batch_size, args.max_sentence_length)
    avg_length = sum((len(x)-1 for x in test_data)) / len(test_data)
    valid_data = read_from_file(args.valid, char_dict, args.batch_size, args.max_sentence_length)
    train_data = read_from_file(args.train, char_dict, args.batch_size, args.max_sentence_length)
    avg_length = sum((len(x)-1 for x in train_data)) / len(train_data)
    number_of_train_batches = int(len(train_data) / args.batch_size)

    print(char_dict)
    print("number of batches: {}".format(number_of_train_batches))
    print("Average number of character per train sentence: {}".format(round(avg_length, 2)), flush=True)

    sum_loss = 0.0
    sum_chars = 0.0
    min_valid_loss = 9999.0

    PRINT_INFO_INTERVAL = args.batch_size * 500

    # do this forever
    for processed_sentences in range(0, 999999999, args.batch_size):
        # Train
        idx = random.randrange(number_of_train_batches)
        batch = train_data[idx * args.batch_size : (idx + 1) * args.batch_size]

        loss = lstm.loss_of_sentences(batch)

        # unscale for actual value
        loss_value = loss.value()
        sum_loss += loss_value
        sum_chars += len(batch) * (len(batch[0]) - 1) # we don't calculate loss for first element

        loss.backward()
        trainer.update()

        # Validate and sample
        if (processed_sentences % PRINT_INFO_INTERVAL) == 0:
            lstm.lstm.disable_dropout()
            sample(lstm, "")
            sample(lstm, "probably")
            sample(lstm, "so then what is this the first")

            def bpc(n):
                return round(n/math.log(2.0), 2)

            valid_loss = round(calc_avg_loss(valid_data), 2)
            test_loss = round(calc_avg_loss(test_data), 2)
            train_loss = round(sum_loss / sum_chars, 2)
            print("Sentences: {}, Train loss: {} (bpc: {}), "
                  "Valid loss: {}(bpc {}) Test loss {}(bpc {})"
                  .format(processed_sentences,
                          train_loss, bpc(train_loss),
                          valid_loss, bpc(valid_loss),
                          test_loss, bpc(test_loss)),
                  flush=True)
            # this is here on purpose: first training batch does not use dropout though
            if args.dropout == 1:
                lstm.lstm.set_dropout(0.5)

        # save based on valid loss and reset counters
        if (processed_sentences % (PRINT_INFO_INTERVAL * 16)) == 0:
            if valid_loss < min_valid_loss:
                print("Saving Model; Train bpc: {} Valid bpc: {}, Test bpc: {}".format(
                    bpc(train_loss), bpc(valid_loss), bpc(test_loss)
                ), flush=True)
                min_valid_loss = valid_loss
                model.save(args.name + ".model", [lstm])
            elif args.adam == 1:
                print("Restarting Adam")
                trainer = dy.AdamTrainer(model)

            # Reset Counters
            print("reset counters")
            sum_loss, sum_chars = 0.0, 0.0
