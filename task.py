#!/usr/bin/env python3

import sys, os, asyncio, traceback, time
import re
# from concurrent.futures import ThreadPoolExecutor as Executor   # or ProcessPoolExecutor as Executor

from worker_pool import WorkerProcessPool, ErrorMessage

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/marian/build')
import libamunmt as nmt


name = 'SUMMA-PUNCT'      # required by rabbitmq module
# executor = None

from SigSilSegmentation import SigSilSegmentation

# required
def init(args=None):
    # global executor
    # executor = Executor(max_workers=args and args.PARALLEL or 1)
    # init_module()                           # initialize in calling thread
    # # executor.submit(init_module).result()   # initialize in worker thread and wait to complete
    global pool
    pool = WorkerProcessPool(worker_run, init_module, count=args.PARALLEL, heartbeat_pause=args.heartbeat_pause, init_args=(args,))
    pool.start()
    # give some time for workers to start
    time.sleep(5)
    pool.watch_heartbeats(args.restart_timeout, args.refresh, args.max_retries_per_job)


def setup_argparser(parser):
    env = os.environ
    parser.add_argument('--heartbeat-pause', type=int, default=env.get('HEARTBEAT_PAUSE', 10),
            help='pause in seconds between heartbeats (or set env variable HEARTBEAT_PAUSE)')
    parser.add_argument('--refresh', type=int, default=env.get('REFRESH', 5),
            help='seconds between pulse checks (or set env variable REFRESH)')
    parser.add_argument('--restart-timeout', type=int, default=env.get('RESTART_TIMEOUT', 5*60),
            help='max allowed seconds between heartbeats, will restart worker if exceeded (or set env variable RESTART_TIMEOUT)')
    parser.add_argument('--max-retries-per-job', type=int, default=env.get('MAX_RETRIES_PER_JOB', 3),
            help='maximum retries per job (or set env variable MAX_RETRIES_PER_JOB)')


def shutdown():
    # global executor
    # executor.shutdown()
    global pool
    return pool.terminate()


def reset():
    global pool
    pool.reset()


# async def run(segment, loop=None):
#     if not loop:
#         loop = asyncio.get_event_loop()
#     return await loop.run_in_executor(executor, translate, segment)


async def process_message(task_data, loop=None, send_reply=None, metadata=None, reject=None, **kwargs):
    # return {'segments': await run(task_data['segments'], loop)}
    global pool
    async with pool.acquire() as worker:
        return await worker(task_data, send_reply)


async def worker_run(document, partial_result_callback=None, loop=None, heartbeat=None, *args, **kwargs):
    return dict(segments=punctuate(document['segments']))


# --- private ---

def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def init_module(args):
    log('Initialize Punctuation module ...')
    #Dynamic creation of variable
    nmt.bOnePass=bool(args.ONEPASS_SEG == "TRUE")
    if not nmt.bOnePass:
        nmt.init('-c model/config.yaml')
    log('Punctuation worker initialized!')


def interleave(words, punctuation_marks):
    if punctuation_marks[-1] == '<SPACE>':
        punctuation_marks[-1] = '<FULL_STOP>'

    punctuation_marks = punctuation_marks[:len(words)]
    if len(words) > len(punctuation_marks):
        punctuation_marks[:-1] + ['<SPACE>'] * (len(words) - len(punctuation_marks)) + punctuation_marks[-1:]

    punctuation_marks_map = {
        '<FULL_STOP>': '.',
        '<COMMA>': ',',
        '<QUESTION_MARK>': '?',
        '<EXCLAMATION_MARK>': '!',
        '<THREE_DOTS>': '...'
    }

    tokens = []
    for (word, punctuation_mark) in zip(words, punctuation_marks):
        tokens.append(word)

        if punctuation_mark in punctuation_marks_map:
            tokens.append({
                'word': punctuation_marks_map[punctuation_mark],
                'time': word['time'] + word['duration'],
                'duration': 0,
                'confidence': 1,
            })

    return tokens

def punctuate(segments):
    """Two passes punctuation scheme.

       parameters segments: segments received from ASR

       [[{'confidence': 1, 'time': 0.9799999594688416, 'duration': 0.2199999988079071, 'word': 'high'},
         {'confidence': 1, 'time': 1.1999999284744263, 'duration': 0.2800000011920929, 'word': 'school'},
         {'confidence': 1, 'time': 1.4800000190734863, 'duration': 0.4399999976158142, 'word': 'student'}]]

       return a list of list of words with punctuation
              symbols
    """
    #First pass segmentation
    allWords, firstPassSentences = resegment(segments)
    if not allWords:
        return []

    #Rabbitmq was launched with one pass flag
    if nmt.bOnePass:
        return words2ctm(allWords, firstPassSentences)

    #Second pass segmentation
    sigSilSegments, startSegmentIndice = [[]], 0
    for strSentence in firstPassSentences:
        # i.e. ['<SPACE> <SPACE> <FULL_STOP>']
        segment_pm = nmt.translate([strSentence.upper()])[0]
        #log(strSentence)
        #log(segment_pm)
        startSegmentIndice, segment = sentenceWordsExtraction(strSentence,
                                                              startSegmentIndice,
                                                              allWords)

        # Punctuated segments
        sigSilSegments[0].extend(interleave(segment, segment_pm.split()))

    return sigSilSegments

def resegment(segments):
    """Significant silence segmentation.

       return allWords           : list of dictionaries for each word
                                   [{'confidence': 1, 'time': 0.9799999594688416, 'duration': 0.2199999988079071, 'word': 'high'}, ...]
              firstPassSentences : a list of sentences [u'w1 w2', u'w3 w4 w5']
    """
    allWords = [w for segment in segments for w in segment]
    if not allWords:
        return [], []
    sss = SigSilSegmentation(cdfThreshold=0.97)

    return allWords, sss.significantSilenceSegmentation(allWords)

def sentenceWordsExtraction(strSentence, startSegmentIndice, allWords):
    """Extract words dictionaries for 'strSentence'.

       return a list of dictionaries, one for each word
    """
    # Sentence words extraction
    wordsList = re.split(u" +", strSentence)
    endSegmentIndice = startSegmentIndice + len(wordsList)
    assert endSegmentIndice <= len(allWords)
    segment = allWords[startSegmentIndice:endSegmentIndice]
    return endSegmentIndice, segment

def words2ctm(allWords, firstPassSentences):
    """Transform sentences to CTM list.

       param allWords          : list of dictionaries for each word
                                  [{'confidence': 1, 'time': 0.9799999594688416, 'duration': 0.2199999988079071, 'word': 'high'}, ...]
             firstPassSentences: list of string for each sentence
                                 [u'sentence 1', u'sentence 2', ...]
       return a list with one segment of word ctms
              [[{}, {}]]
    """
    tokens, startSegmentIndice = [[]], 0
    for strSentence in firstPassSentences:
        #New start and list of dictionaries for sentence
        startSegmentIndice, segment = sentenceWordsExtraction(strSentence,
                                                              startSegmentIndice,
                                                              allWords)
        word = segment[-1]

        #Punctuation mark
        segment.append({
                'word': '.',
                'time': word['time'] + word['duration'],
                'duration': 0,
                'confidence': 1,
            })

        tokens[0].extend(segment)

    return tokens


if __name__ == "__main__":

    import json
    import argparse

    parser = argparse.ArgumentParser(description='Punctuation Task', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--parallel', '-n', dest='PARALLEL', type=int, default=os.environ.get('PARALLEL',1),
            help='messages to process in parallel (or set env variable PARALLEL)')
    parser.add_argument('filename', type=str, default='test.json', nargs='?', help='JSON file with task data')

    setup_argparser(parser)

    args = parser.parse_args()

    print('Reading', args.filename)
    with open(args.filename, 'r') as f:
        task_data = json.load(f)

    init(args)

    try:
        loop = asyncio.get_event_loop()
        # loop.set_debug(True)
        result = loop.run_until_complete(process_message(task_data, loop))
        print('Result:')
        print(result)
    except KeyboardInterrupt:
        print('INTERRUPTED')
    except:
        print('EXCEPTION')
        traceback.print_exc()
        # raise
    finally:
        shutdown()
