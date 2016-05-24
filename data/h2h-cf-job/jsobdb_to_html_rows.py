#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import json

with open(sys.argv[1], 'r') as r, open(sys.argv[2], 'w') as w: 
    rows = json.load(r)
    header = rows[0].keys()
    w.write('<table class="table table-striped">\t<thead>\n\t<tr>\n')
    for c in header:
        w.write('\t\t<th>%s</th>\n' % c)
    w.write('\t</tr>\n</thead>\n\t<tbody class="searchable">\n')
    for row in rows:
        w.write('\t<tr>\n')
        for c in header:
            w.write('\t\t<td>%s</td>\n' % row[c])
        w.write('\t</tr>\n')

    w.write('\t</tbody>\n</table>')
