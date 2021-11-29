class BatchDatasetA(object):
    def __init__(self):
        self.wiki_words_train_file = 'file1.txt'
        self.wiki_hyp_train_file = 'file2.txt'
        self.train_data_source = 'wiki-canonical'
        self.num_passes_wiki_words = 1
        self.total_num_passes_wiki_words = 10

        self.wiki_words_it = open(self.wiki_words_train_file, 'r')
        self.wiki_hyp_it = open(self.wiki_hyp_train_file, 'r')

    def read_one_line(self):
        if self.train_data_source == 'wiki-canonical':
            line = self.wiki_words_it.readline()
        else:
            assert self.train_data_source == 'wiki-canonical-hyperlinks'
            line = self.wiki_hyp_it.readline()

        if not line:
            if self.num_passes_wiki_words == self.total_num_passes_wiki_words:
                self.train_data_source = 'wiki-canonical-hyperlinks'
                self.wiki_words_it.close()
                print('\n\n' + 'Start training on Wiki Hyperlinks' + '\n\n')

            print('Training file is done. Num passes = ', self.num_passes_wiki_words, '. Reopening.')
            self.num_passes_wiki_words += 1

            if self.train_data_source == 'wiki-canonical':
                self.wiki_words_it = open(self.wiki_words_train_file, 'r')
                line = self.wiki_words_it.readline()
            else:
                self.wiki_hyp_it = open(self.wiki_hyp_train_file, 'r')
                line = self.wiki_hyp_it.readline()

        return line

    def __del__(self):
        self.wiki_words_it.close()
        self.wiki_hyp_it.close()

    def patch_of_lines(self, batch_size):
        lines = []
        cnt = 0

        assert batch_size > 0

        while cnt < batch_size:
            # **YD** "read_one_line", not implemented
            line = self.read_one_line()
            cnt += 1
            lines.append(line)

        assert len(lines) == batch_size

        return lines


if __name__ == '__main__':
    batch_dataset_a = BatchDatasetA()
    num_batches_per_epoch = 100
    batch_size = 2
    for batch_index in range(num_batches_per_epoch):
        lines = batch_dataset_a.patch_of_lines(batch_size)
        print(lines)
