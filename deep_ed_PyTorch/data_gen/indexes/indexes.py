import os

# basic_data/wiki_redirects.txt
# wikipedia-link \t  wikipedia-redirected-link

# basic_data/wiki_disambiguation_pages.txt
# wiki-id \t wikipedia-link


class WikiDisambiguationIndex:
    def __init__(self, args):
        self.disambiguation_file = os.path.join(args.root_data_dir, 'basic_data/wiki_disambiguation_pages.txt')
        assert os.path.isfile(self.disambiguation_file)

        self.wiki_disambiguation_index = dict()

        print('==> Loading disambiguation index')
        with open(self.disambiguation_file, 'r') as reader:
            for line in reader:
                parts = line.rstrip('\n').split('\t')
                self.wiki_disambiguation_index[int(parts[0])] = 1

        assert 579 in self.wiki_disambiguation_index
        assert 41535072 in self.wiki_disambiguation_index
        print('    Done loading disambiguation index')

    @property
    def dict(self):
        return self.wiki_disambiguation_index

    def is_disambiguation(self, wikiid):
        return wikiid in self.wiki_disambiguation_index


class WikiRedirectsPagesIndex:
    def __init__(self, args):

        self.redirect_file = os.path.join(args.root_data_dir, 'basic_data/wiki_redirects.txt')
        assert os.path.isfile(self.redirect_file)

        self.wiki_redirects_index = dict()

        print('==> Loading redirects index')
        with open(self.redirect_file, 'r') as reader:
            for line in reader:
                parts = line.rstrip('\n').split('\t')
                self.wiki_redirects_index[parts[0]] = parts[1]

        assert self.wiki_redirects_index['Coercive'] == 'Coercion'
        assert self.wiki_redirects_index['Hosford, FL'] == 'Hosford, Florida'
        print('    Done loading redirects index')

    @property
    def dict(self):
        return self.wiki_redirects_index

    def get_redirected_ent_title(self, ent_name):
        return self.wiki_redirects_index.get(ent_name, ent_name)
