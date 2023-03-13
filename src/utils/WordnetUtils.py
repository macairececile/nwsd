from ufsac.common.POSConverter import POSConverter
from ufsac.common.WordnetHelper import WordnetHelper
from ufsac.ufsac.core.Corpus import Corpus


class WordnetUtils:
    @staticmethod
    def extract_lemma_from_sense_key(sense_key):
        return sense_key[0:sense_key.find("%")]

    @staticmethod
    def extract_pos_from_sense_key(sense_key):
        return POSConverter.to_wn_pos(int(sense_key[sense_key.find("%") + 1:sense_key.find("%") + 2]))

    @staticmethod
    def get_unique_synset_keys_from_sense_keys(wn, sense_keys):
        synset_keys = set()
        for sense_key in sense_keys:
            synset_keys.add(wn.get_synset_key_from_sense_key(sense_key))
        return synset_keys

    @staticmethod
    def get_hypernym_hierarchy(wn, synset_key, hypernymy_hierarchy=None):
        if hypernymy_hierarchy is None:
            hypernymy_hierarchy = []
            WordnetUtils.get_hypernym_hierarchy(wn, synset_key, hypernymy_hierarchy)
            return hypernymy_hierarchy
        else:
            if synset_key in hypernymy_hierarchy:
                return
            hypernymy_hierarchy.append(synset_key)
            hypernym_synset_keys = wn.get_hypernym_synset_keys_from_synset_key(synset_key)
            if len(hypernym_synset_keys) != 0:
                WordnetUtils.get_hypernym_hierarchy(wn, hypernym_synset_keys[0], hypernymy_hierarchy)

    @staticmethod
    def get_hypernym_hierarchy_include_instance_hypernyms(wn, synset_key, hypernymy_hierarchy=None):
        if hypernymy_hierarchy is None:
            hypernymy_hierarchy = []
            WordnetUtils.get_hypernym_hierarchy_include_instance_hypernyms(wn, synset_key, hypernymy_hierarchy)
            return hypernymy_hierarchy
        else:
            if synset_key in hypernymy_hierarchy:
                return
            hypernymy_hierarchy.append(synset_key)
            hypernym_synset_keys = wn.get_hypernym_synset_keys_from_synset_key(synset_key)
            hypernym_synset_keys.append(wn.get_instance_hypernym_synset_keys_from_synset_key(synset_key))
            if len(hypernym_synset_keys) != 0:
                WordnetUtils.get_hypernym_hierarchy(wn, hypernym_synset_keys[0], hypernymy_hierarchy)

    @staticmethod
    def get_reduced_synset_keys_with_hypernyms1(wn, corpora, remove_monosemics, remove_coarse_grained):
        # PAS COMPLET
        sense_tag = "wn" + wn.get_version() + "_key"
        all_vocabulary = {}
        all_hypernym_hierarchy = {}

        for corpus in corpora:
            reader = Corpus.load_from_xml(corpus)

        necessary_synset_keys = set()
        for word_key in all_vocabulary.keys():
            for synset_key in all_vocabulary[word_key]:
                hypernym_hierarchy = all_hypernym_hierarchy[synset_key]
                where_to_stop = len(hypernym_hierarchy)
                found = False
                for i in range(len(hypernym_hierarchy)):
                    if found:
                        break
                    for synset_key2 in all_vocabulary[word_key]:
                        if synset_key2 == synset_key:
                            continue
                        if found:
                            break
                        hypernym_hierarchy2 = all_hypernym_hierarchy[synset_key2]
                        for j in range(len(hypernym_hierarchy2)):
                            if hypernym_hierarchy[i] == hypernym_hierarchy2[j]:
                                where_to_stop = i
                                found = True
                                break
                if where_to_stop == 0:
                    necessary_synset_keys.add(hypernym_hierarchy[0])
                else:
                    necessary_synset_keys.add(hypernym_hierarchy[where_to_stop - 1])

        synset_keys_to_simple_synset_key = {}
        for synset_key in all_hypernym_hierarchy.keys():
            hypernym_hierarchy = all_hypernym_hierarchy[synset_key]
            for i in range(len(hypernym_hierarchy)):
                if hypernym_hierarchy[i] in necessary_synset_keys:
                    synset_keys_to_simple_synset_key[synset_key] = hypernym_hierarchy[i]
                    break

        return synset_keys_to_simple_synset_key

    @staticmethod
    def read_word(self, word, sense_tag, remove_monosemics, wn, remove_coarse_grained, all_vocabulary,
                  all_hypernym_hierarchy):
        if word.has_annotation(sense_tag):
            word_key = word.get_annotation_value("lemma") + "%" + POSConverter.to_wn_pos(
                word.get_annotation_value("pos"))
            if remove_monosemics and len(wn.get_sense_key_list_from_word_key(word_key)) == 1:
                return
            sense_keys = word.get_annotation_values(sense_tag, ";")
            synset_keys = WordnetUtils.get_unique_synset_keys_from_sense_keys(wn, sense_keys)
            if remove_coarse_grained and len(synset_keys) > 1:
                return
            for synset_key in synset_keys:
                if word_key not in all_vocabulary.keys():
                    all_vocabulary[word_key] = set()
                all_vocabulary[word_key] = synset_key
                if synset_key not in all_hypernym_hierarchy.keys():
                    all_hypernym_hierarchy[synset_key] = self.get_hypernym_hierarchy(wn, synset_key)

    @staticmethod
    def get_reduced_synset_keys_with_hypernyms2(self, wn, corpora, remove_monosemics, remove_coarse_grained):
        # PAS COMPLET
        sense_tag = "wn" + wn.get_version() + "_key"
        all_vocabulary = {}
        all_hypernym_hierarchy = {}

        for corpus in corpora:
            reader = Corpus.load_from_xml(corpus)

        synset_keys_to_simple_synset_key = {}
        for word_key in all_vocabulary.keys():
            for synset_key in all_vocabulary[word_key]:
                hypernym_hierarchy = all_hypernym_hierarchy[synset_key]
                where_to_stop = len(hypernym_hierarchy)
                found = False
                for i in range(len(hypernym_hierarchy)):
                    if found:
                        break
                    for synset_key2 in all_vocabulary[word_key]:
                        if synset_key2 == synset_key:
                            continue
                        if found:
                            break
                        hypernym_hierarchy2 = all_hypernym_hierarchy[synset_key2]
                        for j in range(len(hypernym_hierarchy2)):
                            if hypernym_hierarchy[i] == hypernym_hierarchy2[j]:
                                where_to_stop = i
                                found = True
                                break
                if where_to_stop == 0:
                    synset_keys_to_simple_synset_key[word_key + synset_key] = hypernym_hierarchy[0]
                else:
                    synset_keys_to_simple_synset_key[word_key + synset_key] = hypernym_hierarchy[where_to_stop - 1]
        return synset_keys_to_simple_synset_key

    @staticmethod
    def get_reduced_synset_keys_with_hypernyms3(wn):
        all_vocabulary = {}
        all_hypernym_hierarchy = {}
        for word_key in wn.get_vocabulary():
            if word_key not in all_vocabulary:
                all_vocabulary[word_key] = set()
            for sense_key in wn.get_sense_key_list_from_word_key(word_key):
                synset_key = wn.get_synset_key_from_sense_key(sense_key)
                all_vocabulary[word_key].add(synset_key)
                if synset_key not in all_hypernym_hierarchy:
                    all_hypernym_hierarchy[synset_key] = WordnetUtils.get_hypernym_hierarchy(wn, synset_key)

        necessary_synset_keys = set()
        for word_key in all_vocabulary.keys():
            for synset_key in all_vocabulary[word_key]:
                hypernym_hierarchy = all_hypernym_hierarchy[synset_key]
                where_to_stop = len(hypernym_hierarchy)
                found = False
                for i in range(len(hypernym_hierarchy)):
                    if found:
                        break
                    for synset_key2 in all_vocabulary[word_key]:
                        if synset_key == synset_key2:
                            continue
                        if found:
                            break
                        hypernym_hierarchy2 = all_hypernym_hierarchy[synset_key2]
                        for j in range(len(hypernym_hierarchy2)):
                            if hypernym_hierarchy[i] == hypernym_hierarchy2[j]:
                                where_to_stop = i
                                found = True
                                break
                if where_to_stop == 0:
                    necessary_synset_keys.add(hypernym_hierarchy[0])
                else:
                    necessary_synset_keys.add(hypernym_hierarchy[where_to_stop - 1])

        synset_keys_to_simple_synset_key = {}
        for synset_key in all_hypernym_hierarchy.keys():
            hypernym_hierarchy = all_hypernym_hierarchy[synset_key]
            for i in range(len(hypernym_hierarchy)):
                if hypernym_hierarchy[i] in necessary_synset_keys:
                    synset_keys_to_simple_synset_key[synset_key] = hypernym_hierarchy[i]
                    break

        return synset_keys_to_simple_synset_key

    @staticmethod
    def get_reduced_synset_keys_with_hypernyms4(wn):
        all_vocabulary = {}
        all_hypernym_hierarchy = {}

        for word_key in wn.get_vocabulary():
            if word_key not in all_vocabulary.keys():
                all_vocabulary[word_key] = set()
            for sense_key in wn.get_sense_key_list_from_word_key(word_key):
                synset_key = wn.get_synset_key_from_sense_key(sense_key)
                all_vocabulary[word_key].add(synset_key)
                if synset_key not in all_hypernym_hierarchy.keys():
                    all_hypernym_hierarchy[synset_key] = WordnetUtils.get_hypernym_hierarchy(wn, synset_key)

        synset_keys_to_simple_synset_key = {}
        for word_key in all_vocabulary.keys():
            for synset_key in all_vocabulary[word_key]:
                hypernym_hierarchy = all_hypernym_hierarchy[synset_key]
                where_to_stop = len(hypernym_hierarchy)
                found = False
                for i in range(0, len(hypernym_hierarchy)):
                    if found:
                        break
                    for synset_key2 in all_vocabulary[word_key]:
                        if synset_key2 == synset_key:
                            continue
                        if found:
                            break
                        hypernym_hierarchy2 = all_hypernym_hierarchy[synset_key2]
                        for j in range(0, len(hypernym_hierarchy2)):
                            if hypernym_hierarchy[i] == hypernym_hierarchy2[j]:
                                where_to_stop = i
                                found = True
                                break
                if where_to_stop == 0:
                    synset_keys_to_simple_synset_key[word_key + synset_key] = hypernym_hierarchy[0]
                else:
                    synset_keys_to_simple_synset_key[word_key + synset_key] = hypernym_hierarchy[where_to_stop - 1]
        return synset_keys_to_simple_synset_key

    @staticmethod
    def get_sense_compression_through_hypernyms_and_instance_hypernyms_clusters(wn, current_clusters=None):
        all_vocabulary = {}
        all_hypernym_hierarchy = {}

        for word_key in wn.get_vocabulary():
            if word_key not in all_vocabulary.keys():
                all_vocabulary[word_key] = set()
            for sense_key in wn.get_sense_key_list_from_word_key(word_key):
                synset_key = wn.get_synset_key_from_sense_key(sense_key)
                all_vocabulary[word_key].add(synset_key)
                if synset_key not in all_hypernym_hierarchy.keys():
                    all_hypernym_hierarchy[synset_key] = WordnetUtils.get_hypernym_hierarchy_include_instance_hypernyms(
                        wn,
                        synset_key)

        necessary_synset_keys = set()
        for word_key in all_vocabulary.keys():
            for synset_key in all_vocabulary[word_key]:
                hypernym_hierarchy = all_hypernym_hierarchy[synset_key]
                where_to_stop = len(hypernym_hierarchy)
                found = False
                for i in range(len(hypernym_hierarchy)):
                    if found:
                        break
                    for synset_key2 in all_vocabulary[word_key]:
                        if synset_key2 == synset_key:
                            continue
                        if found:
                            break
                        hypernym_hierarchy2 = all_hypernym_hierarchy[synset_key2]
                        for j in range(len(hypernym_hierarchy2)):
                            if hypernym_hierarchy[i] == hypernym_hierarchy2[j]:
                                where_to_stop = i
                                found = True
                                break
                if where_to_stop == 0:
                    necessary_synset_keys.add(hypernym_hierarchy[0])
                else:
                    necessary_synset_keys.add(hypernym_hierarchy[where_to_stop - 1])

        synset_keys_to_simple_synset_key = {}
        for synset_key in all_hypernym_hierarchy.keys():
            hypernym_hierarchy = all_hypernym_hierarchy[synset_key]
            for i in range(len(hypernym_hierarchy)):
                if hypernym_hierarchy[i] in necessary_synset_keys:
                    synset_keys_to_simple_synset_key[synset_key] = hypernym_hierarchy[i]
                    break

        return synset_keys_to_simple_synset_key

    @staticmethod
    def get_sense_compression_through_hypernyms_clusters(wn=None, current_clusters=None):
        if wn is None:
            return WordnetUtils.get_reduced_synset_keys_with_hypernyms3(WordnetHelper.wn30())
        else:
            return WordnetUtils.get_reduced_synset_keys_with_hypernyms3(wn)

    @staticmethod
    def get_sense_compression_through_antonyms_clusters(wn, current_clusters):
        antonym_clusters = {}
        for synset_key in current_clusters.values():
            if synset_key in antonym_clusters:
                continue
            antonym_synset_keys = wn.get_antonym_synset_keys_from_synset_key(synset_key)
            for antonym_synset_key in antonym_synset_keys:
                antonym_clusters[antonym_synset_key] = synset_key
            antonym_clusters[synset_key] = synset_key
        new_clusters = {}
        for synset_key in current_clusters.keys():
            new_clusters[synset_key] = antonym_clusters[current_clusters[synset_key]]
        return new_clusters

    @staticmethod
    def init_mapping():
        wn = WordnetHelper.wn30()
        mapping = {}
        for word_key in wn.get_vocabulary():
            for sense_key in wn.get_sense_key_list_from_word_key(word_key):
                synset_key = wn.get_synset_key_from_sense_key(sense_key)
                if synset_key not in mapping.keys():
                    mapping[synset_key] = synset_key
        return mapping

    @staticmethod
    def check_is_okay_to_change_in_mapping(key, new_value, mapping, synset_key_to_word_keys, word_key_to_synset_keys):
        for wordKey in synset_key_to_word_keys[key]:
            set_str = set()
            for synset_key in word_key_to_synset_keys[wordKey]:
                if key == synset_key:
                    mapped_synset_key = new_value
                else:
                    mapped_synset_key = mapping[synset_key]
                if mapped_synset_key in set_str:
                    return False
                set_str.add(mapped_synset_key)
        return True

    @staticmethod
    def check_is_okay_to_merge_clusters(cluster_key1, cluster_key2, inverse_mapping, mapping, synset_key_to_word_keys,
                                        word_key_to_synset_keys):
        for synset_key in inverse_mapping[cluster_key2]:
            if not WordnetUtils.check_is_okay_to_change_in_mapping(synset_key, cluster_key1, mapping,
                                                                   synset_key_to_word_keys,
                                                                   word_key_to_synset_keys):
                return False
        return True

    @staticmethod
    def get_sense_compression_through_all_relations_clusters():
        wn = WordnetHelper.wn30()
        synset_key_to_word_keys = {}
        for word_key in wn.get_vocabulary():
            for sense_key in wn.get_sense_key_list_from_word_key(word_key):
                synset_key = wn.get_synset_key_from_sense_key(sense_key)
                if synset_key not in synset_key_to_word_keys.keys():
                    synset_key_to_word_keys[synset_key] = []
                    synset_key_to_word_keys[synset_key].append(word_key)

        word_key_to_synset_keys = {}
        for word_key in wn.get_vocabulary():
            word_key_to_synset_keys[word_key] = []
            for sense_key in wn.get_sense_key_list_from_word_key(word_key):
                synset_key = wn.get_synset_key_from_sense_key(sense_key)
                word_key_to_synset_keys[word_key].append(synset_key)

        mapping = WordnetUtils.init_mapping()
        inverse_mapping = {}
        for synset_key in mapping.keys():
            inverse_mapping[synset_key] = [synset_key]

        related_clusters = {}
        for synset_key in mapping.keys():
            related_clusters[synset_key] = [wn.get_related_synsets_key_from_synset_key(synset_key)]
            if len(wn.get_sense_key_list_from_synset_key(synset_key)) == 1:
                related_clusters[synset_key].extend(
                    wn.get_related_synsets_key_from_sense_key(wn.get_sense_key_list_from_synset_key(synset_key)[0]))

        for synset_key in mapping.keys():
            for related_synset_key in related_clusters[synset_key]:
                related_clusters[related_synset_key].append(synset_key)

        for synset_key in mapping.keys():
            if related_clusters[synset_key] != synset_key:
                related_clusters[synset_key] = [synset_key]

        cluster_sizes = {}
        for synset_key in mapping.keys():
            cluster_sizes[synset_key] = 1

        previous_total = len(inverse_mapping)
        total = -1
        step = 0
        while total != previous_total:
            print("sense cluster creation step " + str(step))
            step += 1
            previous_total = len(inverse_mapping)
            iter = sorted(cluster_sizes.items(), key=lambda x: x[1])
            iter = [i[0] for i in iter]
            for cluster_key in iter:
                iter2 = sorted(related_clusters[cluster_key], key=lambda x: cluster_sizes[x])
                for related_cluster_key in iter2:
                    if WordnetUtils.check_is_okay_to_merge_clusters(cluster_key, related_cluster_key, inverse_mapping,
                                                                    mapping,
                                                                    synset_key_to_word_keys, word_key_to_synset_keys):
                        for related_synset_key in inverse_mapping[related_cluster_key]:
                            mapping[related_synset_key] = cluster_key
                        inverse_mapping[cluster_key].append(inverse_mapping[related_cluster_key])
                        inverse_mapping.pop(related_cluster_key)
                        for related_related_clusters in related_clusters[related_cluster_key]:
                            if cluster_key == related_related_clusters:
                                continue
                            if related_related_clusters not in related_clusters[cluster_key]:
                                related_clusters[cluster_key].append(related_related_clusters)
                                related_clusters[related_related_clusters].append(cluster_key)
                            related_clusters[related_related_clusters].remove(related_cluster_key)
                        related_clusters[cluster_key].remove(related_cluster_key)
                        related_clusters.pop(related_cluster_key)

                        cluster_sizes[cluster_key] = cluster_sizes[cluster_key] + cluster_sizes[related_cluster_key]
                        cluster_sizes.pop(related_cluster_key)

                        total = len(inverse_mapping)
                        print(" - size is " + str(total))
                        continue
        return mapping

    @staticmethod
    def get_sense_compression_clusters(wn, hypernyms, instance_hypernyms, antonyms):
        clusters = {}
        for word_key in wn.get_vocabulary():
            for sense_key in wn.get_sense_key_list_from_word_key(word_key):
                synset_key = wn.get_synset_key_from_sense_key(sense_key)
                if sense_key not in clusters.keys():
                    clusters[synset_key] = synset_key
        if hypernyms:
            if instance_hypernyms:
                clusters = WordnetUtils.get_sense_compression_through_hypernyms_and_instance_hypernyms_clusters(wn,
                                                                                                                clusters)
            else:
                clusters = WordnetUtils.get_sense_compression_through_hypernyms_clusters(wn, clusters)
        if antonyms:
            clusters = WordnetUtils.get_sense_compression_through_antonyms_clusters(wn, clusters)
        return clusters

    @staticmethod
    def get_sense_compression_clusters_from_file(file_path):
        try:
            mapping = {}
            reader = open(file_path, 'r')
            for line in reader.readlines():
                line = line.split(" ")
                mapping[line[0]] = line[1].rstrip()
            return mapping
        except Exception as e:
            RuntimeError(e)
