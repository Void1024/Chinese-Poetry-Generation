from generator import Generator


topics = [
    ['明月','长夜','孤舟','楼阁','思'],
    ['明月','长夜','孤舟','楼阁','悲'],
    ['明月','长夜','孤舟','楼阁','乐'],
    ['星河','流光','山川','长天','思'],
    ['星河','流光','山川','长天','悲'],
    ['星河','流光','山川','长天','乐'],
    ['春风','桃花','杨柳','燕','喜'],
    ['春风','桃花','杨柳','燕','悲'],
    ['春风','桃花','杨柳','燕','乐'],
    ['沙场','龙城','铁马','金戈','怒'],
]

def gen_poems():
    g = Generator()
    with open('poems.txt', 'w', encoding = 'utf-8') as fout:
        for topic in topics:
            poem = g.generate(topic[:4], topic[4])
            poetry = ','.join(poem)
            fout.write(poetry + '\n')

if __name__ == '__main__':
    gen_poems()