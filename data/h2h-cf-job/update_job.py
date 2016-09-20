#!/usr/bin/env python2
# -*- coding: utf-8 -*-

if __name__ == "__main__":
    import sys
    sys.path.append('../..')
    from generate_dialog_tasks import Dialog
    from e2end.dataset.dstc2 import Dstc2DB
import crowdflower


db = Dstc2DB('../../data/dstc2/data.dstc2.db.json')
data = Dialog.load_answers('./for-next-round/empty10b.csv', db)
data = data.df.to_dict(orient='records')

title = 'Chat-DO-NOT-USE'
instructions = open('./CFJOB-reply.instructions.html').read()
cml = open('./CFJOB-reply.html').read()  # Ignoring 406 "Not Accepted" error: CrowdFlowerError: 414 Request-URI Too Large at https://api.crowdflower.com/v1/jobs/913237


jscript = open('./CFJOB-reply.js').read()
css = open('./CFJOB-reply.css').read()

conn = crowdflower.Connection()
job = None
for j in conn.jobs():
    t = j.properties['title']
    if t == title:
        job = j

if not job:
    raise ValueError('Not find the job')


conn = crowdflower.Connection()  # using CROWDFLOWER_API_KEY variable from shell
job = conn.upload(data)

# options = { 'front_load': 1,}  # quiz mode = 1; turn off with 0
options = {}
update_result = job.update({
    'title': title,
    'included_countries': ['US', 'GB', 'CZ'],  
    # Limit to the USA and United Kingdom
    # Please note, if you are located in another country and you would like
    # to experiment with the sandbox (internal workers) then you also need
    # to add your own country. Otherwise your submissions as internal worker
    # will be rejected with Error 301 (low quality).
    'payment_cents': 1,
    'judgments_per_unit': 1,
    'instructions': instructions,
    'cml': cml,
    'css': css,
    'js': jscript,
    # 'options': options,
})
if 'errors' in update_result:
    print(update_result['errors'])

# job.gold_add('gender', 'gender_gold')

# job.launch(3, channels=['cf_internal'])
# job.cancel
# job.ping
