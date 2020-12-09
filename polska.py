import string
import math

prefixes = ['sin', 'sqrt', 'cos', 'tan', '~']
postfixes = ['!']
infixes = ['+', '-', '*', ':', '^', '/']
prioritets = [('+', 3), ('-', 3), ('*', 2), (':', 2),
              ('/', 2), ('^', 1), ('(', 4), (')', 4)]


def get_prior(c):
    for i in prioritets:
        if c == i[0]:
            return i[1]
    return 0


def fac(n):
    if n == 0:
        return 1
    return fac(n-1) * n


def contains(c, list):
    for i in list:
        if i in c:
            return True
    return False


def polska(o):
    if len(o) == 0:
        return
    new_o = []
    # print(o)
    t = 0
    for i in o:
        if i == '-' and (t == 0 or (t+1 < len(new_o) and new_o[t+1] == '(')):
            new_o.append('~')
        else:
            if len(new_o) == 0:
                new_o.append(i)
            else:
                if new_o[-1].isnumeric() and i.isnumeric():
                    new_o[-1] = str(int(new_o[-1])*10 + int(i))
                else:
                    new_o.append(i)
        t += 1
    # print(new_o)
    output = []
    stack = []
    for i in new_o:
        if i.isnumeric() or i == "pi":
            if i == "pi":
                output.append(str(math.pi))
            else:
                output.append(i)
        elif contains(i, postfixes):
            output.append(i)
        elif contains(i, prefixes):
            stack.append(i)
        elif i == '(':
            stack.append(i)
        elif i == ')':
            while True:
                if len(stack) == 0:
                    print("error of math")
                    return
                j = stack.pop()
                if j == '(':
                    break
                else:
                    output.append(j)
        elif contains(i, infixes):
            if len(stack) == 0:
                stack.append(i)
            else:
                j = stack[-1]
                if get_prior(j) <= get_prior(i):
                    while len(stack) != 0:
                        if get_prior(stack[-1]) <= get_prior(i):
                            output.append(stack.pop())
                        else:
                            break
                    stack.append(i)
                else:
                    stack.append(i)
    while len(stack) != 0:
        output.append(stack.pop())
    # print(output)
    stack = []
    for i in output:
        if contains(i, infixes):
            b = float(stack.pop())
            a = float(stack.pop())
            if i == '+':
                stack.append(str(a + b))
            elif i == '-':
                stack.append(str(a - b))
            elif i == '/' or i == ':':
                stack.append(str(a / b))
            elif i == '*':
                stack.append(str(a*b))
            elif i == '^':
                stack.append(str(pow(a, b)))
        elif contains(i, prefixes):
            a = float(stack.pop())
            print(a)
            if 'sqrt' == i:
                stack.append(str(math.sqrt(a)))
            elif 'sin' == i:
                stack.append(str(math.sin(a)))
            elif 'cos' == i:
                stack.append(str(math.cos(a)))
            elif 'tan' == i:
                stack.append(str(math.tan(a)))
            elif 'sqrt' in i:
                n = i[4:]
                stack.append(str(pow(a, 1./int(n))))
            elif '~' in i:
                stack.append(str(-a))
        elif contains(i, postfixes):
            n = stack.pop()
            a = int(round(float(n)))
            if '!' == i:
                stack.append(str(fac(a)))
        else:
            stack.append(i)
    #print('=' + stack.pop())
    return round(float(stack.pop()), 6)


#print(polska(['sqrt3', '(', '20', '+', '7', ')', '+', 'sqrt2', '(', '3', '*', '3', ')']))
