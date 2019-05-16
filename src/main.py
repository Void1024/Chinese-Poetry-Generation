#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from generator import Generator
from plan import Planner



if __name__ == '__main__':
    planner = Planner()
    generator = Generator()
    while True:
        hints = input("Type in hints >> ")
        input_list = hints.split('#')

        keywords = planner.plan(input_list[0])
        emotion = '悲' if len(input_list) < 2 else input_list[1]
        print("Keywords: " + ' '.join(keywords))
        poem = generator.generate(keywords, emotion[0])
        print("Poem generated:")
        for sentence in poem:
            print(sentence)


