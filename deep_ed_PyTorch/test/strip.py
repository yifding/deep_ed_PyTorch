import re
#def cus_strip(s):
#    return s.replace("^%s*(.-)%s*$", "%1")


def cus_strip(s):

    #return re.sub("^\s*(.*)\s*$", '', s)
    return s.strip('\t\n\r\f')

print(repr(cus_strip('\u00a0 New Jersey  ')))