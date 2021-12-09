import json
import argparse
from deep_ed_PyTorch.entities.ent_name2id_freq import EntNameID
# **YD** the logic of extract_text_and_hyp does not consider line changing and may delete too much line.


class ParseWikiDumpTool:
    def __init__(self, args):
        self.args = args
        self.ent_name_id = EntNameID(self.args)

    def extract_page_entity_title(self, line):
        doc_str = '<doc id="'
        startoff = line.find(doc_str)
        assert startoff >= 0, 'not find <doc id=" symbol in the line'
        endoff = startoff + len(doc_str)
        startquotes = line.find('"', endoff)
        ent_wikiid = int(line[endoff: startquotes])

        title_str = ' title="'
        starttitlestartoff = line.find(title_str)
        assert starttitlestartoff >= 0, 'not find  title=" symbol in the line'
        starttitleendoff = starttitlestartoff + len(title_str)
        endtitleoff = line.find('">')
        ent_name = line[starttitleendoff: endtitleoff]

        if ent_wikiid != self.ent_name_id.get_ent_wikiid_from_name(ent_name, True):
            return self.ent_name_id.get_ent_wikiid_from_name(ent_name, True)
        else:
            return ent_wikiid

    def extract_text_and_hyp(self, line, mark_mentions=False):
        hyp_anchor = '<a href="'
        hyp_mid = '">'
        hyp_end = '</a>'

        # **YD** not sure the set usage
        list_hyp = dict()
        text = ''
        list_ent_errors = 0
        parsing_errors = 0
        disambiguation_ent_errors = 0
        diez_ent_errors = 0

        end_end_hyp = 0
        begin_end_hyp = 0
        begin_start_hyp = line.find(hyp_anchor)
        end_start_hyp = begin_start_hyp + len(hyp_anchor) if begin_start_hyp != -1 else -1

        num_mentions = 0

        while begin_start_hyp >= 0:

            text += line[end_end_hyp: begin_start_hyp]
            next_quotes = line.find(hyp_mid, end_start_hyp)
            end_quotes = next_quotes + len(hyp_mid) if next_quotes != -1 else -1

            if next_quotes >= 0:
                ent_name = line[end_start_hyp: next_quotes]
                begin_end_hyp = line.find(hyp_end, end_quotes)
                end_end_hyp = begin_end_hyp + len(hyp_end) if begin_end_hyp != -1 else -1

                if begin_end_hyp >= 0:
                    mention = line[end_quotes: begin_end_hyp]
                    mention_marker = False

                    good_mention = False
                    if mention and len(mention) >=1 and 'wikipedia' not in mention and 'Wikipedia' not in mention:
                        good_mention = True

                    if good_mention:
                        i = ent_name.find('wikt:')
                        if i == 0:
                            ent_name = ent_name[5:]
                        ent_name = self.ent_name_id.preprocess_ent_name(ent_name)

                        i = ent_name.find('List of ')
                        if i == 0:
                            list_ent_errors += 1
                        else:
                            if ent_name.find('#') >= 0:
                                diez_ent_errors += 1
                            else:
                                ent_wikiid = self.ent_name_id.get_ent_wikiid_from_name(ent_name, True)
                                if ent_wikiid == self.ent_name_id.unk_ent_wikiid:
                                    disambiguation_ent_errors += 1
                                else:
                                    # -- A valid(entity, mention) pair
                                    num_mentions += 1

                                    list_hyp[str(num_mentions)] = {
                                        'ent_wikiid': ent_wikiid,
                                        'mention': mention,
                                        'cnt': str(num_mentions),
                                    }

                                    if mark_mentions:
                                        mention_marker = True

                    if not mention_marker:
                        text += ' ' + mention + ' '
                    else:
                        text += ' MMSTART' + str(num_mentions) + ' ' + mention + ' MMEND' + str(num_mentions) + ' '

                else:
                    parsing_errors += 1
                    begin_start_hyp = -1

            else:
                parsing_errors += 1
                begin_start_hyp = -1

            if begin_start_hyp >= 0:
                begin_start_hyp = line.find(hyp_anchor, end_start_hyp)
                end_start_hyp = begin_start_hyp + len(hyp_anchor) if begin_start_hyp != -1 else -1

        if end_end_hyp >= 0:
            text += line[end_end_hyp:]
        else:
            if not mark_mentions:
                text = line
            else:
                text = ''
                list_hyp = dict()

        return list_hyp, text, list_ent_errors, parsing_errors, disambiguation_ent_errors, diez_ent_errors


def test(args):
    print('\n Unit tests:')
    parse_wiki_dump_tool = ParseWikiDumpTool(args)

    test_line_1 = '<a href="Anarchism">Anarchism</a> is a <a href="political philosophy">political philosophy</a> that advocates<a href="stateless society">stateless societies</a>often defined as <a href="self-governance">self-governed</a> voluntary institutions, but that several authors have defined as more specific institutions based on non-<a href="Hierarchy">hierarchical</a> <a href="Free association (communism and anarchism)">free associations</a>..<a href="Anarchism">Anarchism</a>'

    test_line_2 = 'CSF pressure, as measured by <a href="lumbar puncture">lumbar puncture</a> (LP), is 10-18 <a href="Pressure#H2O">'

    test_line_3 = 'Anarchism'

    test_line_4 = '<doc id="12" url="http://en.wikipedia.org/wiki?curid=12" title="Anarchism">'

    test_line_5 = 'McDonnell is also a member of the Board of Directors of <a href="The Humane Society of the United States">The Humane Society of the United States</a>, <a href="The Fund for Animals">The Fund for Animals</a> and The <a href="Charles M. Schulz">Charles M. Schulz</a> Museum . McDonnell and his wife Karen reside in <a href="\xa0 New Jersey">\xa0 New Jersey</a>, with their dog Amelie, and their cat, Not Ootie. Their Jack Russell, Earl, who was the inspiration and constant muse for the Mutts character of the same name, died in November 2007 after living with Patrick for over 18 years.\n'
    # test_line_5 = 'Some adaptations of the Latin alphabet are augmented with <a href="ligature (typography)">ligatures</a>, such as <a href="æ">æ</a> in <a href="Old English language">Old English</a> and <a href="Icelandic language">Icelandic</a> and <a href="Ou (letter)">Ȣ</a> in <a href="Algonquian languages">Algonquian</a>; by borrowings from other alphabets, such as the <a href="thorn (letter)">thorn</a> þ in <a href="Old English language">Old English</a> and <a href="Icelandic language">Icelandic</a>, which came from the <a href="Runic alphabet">Futhark</a> runes; and by modifying existing letters, such as the <a href="Eth (letter)">eth</a> ð of Old English and Icelandic, which is a modified "d". Other alphabets only use a subset of the Latin alphabet, such as Hawaiian, and <a href="Italian language">Italian</a>, which uses the letters "j, k, x, y" and "w" only in foreign words.\n'
    test_line_6 = '<a href="æ">æ</a>'

    """
    list_hype, text, list_error, parsing_error, dis_error, diez_error = parse_wiki_dump_tool.extract_text_and_hyp(test_line_1, False)
    print(json.dumps(list_hype, indent=2))
    print(text)
    print(list_error, parsing_error, dis_error, diez_error)
    print('*' * 100)

    list_hype, text, list_error, parsing_error, dis_error, diez_error = parse_wiki_dump_tool.extract_text_and_hyp(test_line_1, True)
    print(json.dumps(list_hype, indent=2))
    print(text)
    print(list_error, parsing_error, dis_error, diez_error)
    print('*' * 100)

    list_hype, text, list_error, parsing_error, dis_error, diez_error = parse_wiki_dump_tool.extract_text_and_hyp(test_line_2, True)
    print(json.dumps(list_hype, indent=2))
    print(text)
    print(list_error, parsing_error, dis_error, diez_error)
    print('*' * 100)

    list_hype, text, list_error, parsing_error, dis_error, diez_error = parse_wiki_dump_tool.extract_text_and_hyp(test_line_3, False)
    print(json.dumps(list_hype, indent=2))
    print(text)
    print(list_error, parsing_error, dis_error, diez_error)
    print('*' * 100)

    print(parse_wiki_dump_tool.extract_page_entity_title(test_line_4))
    """
    list_hype, text, list_error, parsing_error, dis_error, diez_error = \
        parse_wiki_dump_tool.extract_text_and_hyp(test_line_5, False)
    print(json.dumps(list_hype, indent=2))
    print(text)
    print(list_error, parsing_error, dis_error, diez_error)
    print('*' * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='evaluate the coverage percent of entity dictionary on evaluate GT entity'
    )

    parser.add_argument(
        '--root_data_dir',
        type=str,
        # default='/scratch365/yding4/deep_ed_PyTorch/data/',
        required=True,
        help='Root path of the data, $DATA_PATH.',
    )

    args = parser.parse_args()
    test(args)

