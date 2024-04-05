# coding: utf-8
from difflib import SequenceMatcher
from .base import PhoneTypeWriter


def evaluate(phone_type_writer: PhoneTypeWriter, spec_image_data_list, noised=True):
    matcher = SequenceMatcher()

    n_corr = 0 # 正解数
    n_ins  = 0 # 挿入誤り数
    n_del  = 0 # 脱落誤り数
    n_rep  = 0 # 置換誤り数
    
    for spec_image_data in spec_image_data_list:
        images = None
        if noised:
            images = spec_image_data.noised_images
        else:
            images = spec_image_data.clean_images

        predicted_phones = phone_type_writer.predict(images)
        correct_phones = spec_image_data.trans.pron.split(' ')

        matcher.set_seqs(correct_phones, predicted_phones)
        for t, i1, i2, j1, j2 in matcher.get_opcodes():
            if t == 'equal':
                n_corr += j2 - j1
            elif t == 'insert':
                n_ins += j2 - j1
            elif t == 'delete':
                n_del += i2 - i1
            else:
                n_i = i2 - i1
                n_j = j2 - j1
                if n_i > n_j:
                    n_rep += n_j
                    n_del += n_i - n_j
                elif n_i < n_j:
                    n_rep += n_i
                    n_ins += n_j
                else:
                    n_rep += n_i
        # print (' '.join(correct_phones))
        # print (' '.join(predicted_phones))
    correction_rate = n_corr / (n_corr + n_rep + n_del)
    accuracy_rate = (n_corr - n_ins) / (n_corr + n_rep + n_del)                                 
    print (n_corr, n_rep, n_del, n_ins,
           "%.2f" % (100 * correction_rate),
           "%.2f" % (100 * accuracy_rate))
