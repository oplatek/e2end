# coding: utf-8
from generate_dialog_tasks import Dialog
import crowdflower
from IPython import get_ipython


data = Dialog.load_answers('./data/test.csv')
data = data.df.to_dict(orient='records')
print(data)

conn = crowdflower.Connection()
job = None
for j in conn.jobs():
    title = j.properties['title']
    if title == 'Test-Chat':
        job = j

if not job:
    raise ValueError('Not find the job')


conn = crowdflower.Connection()  # using CROWDFLOWER_API_KEY variable from shell
get_ipython().magic('pwd ')
job = conn.upload(data)

cml = open('./CFJOB-reply.html').read()
jscript = open('./CFJOB-reply.js').read()
css = open('./CFJOB-reply.css').read()
instructions = open('./.CFJOB-reply.instructions.html')
update_result = job.update({
    'title': 'test-chat',
    'included_countries': ['US', 'GB', 'CZ'],  
    # Limit to the USA and United Kingdom
    # Please note, if you are located in another country and you would like
    # to experiment with the sandbox (internal workers) then you also need
    # to add your own country. Otherwise your submissions as internal worker
    # will be rejected with Error 301 (low quality).
    'payment_cents': 5,
    'judgments_per_unit': 2,
    'instructions': 'some <i>instructions</i> html',
    'cml': cml,
    # 'css': css,
    # 'javascript': jscript,
    'options': {
        'front_load': 1,  # quiz mode = 1; turn off with 0
    }
})
if 'errors' in update_result:
        print(update_result['errors'])

# job.gold_add('gender', 'gender_gold')

# job.launch(3, channels=['cf_internal'])
# job.cancel
# job.ping
