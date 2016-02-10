class Individual:
    def __init__(self, sex, y1, y2, x1, x2, infected, exposure):
        self.sex = sex
        self.y1 = y1
        self.y2 = y2
        self.x1 = x1
        self.x2 = x2
        #yeah I changed these names, its worth the 5 extra letters to make your code more readable
        #especially "inf" which by default means "infinity" to me
        self.infected = infected
        self.exposure = exposure

    def __str__(self):
        return "Individual(%s, %s, %s, %s, %s, %s, %s)" % (self.sex, self.y1, self.y2, self.x1, self.x2, self.infected, self.exposure)


####example usage

me = Individual(0, 1, 2, 4, 5, 0, 0)

print("sex: %s" % me.sex)
print("My x1 and x2: %s %s" % (me.x1, me.x2))

print(me)

print(me.x1+me.x2)

print("Oh noes, I'm infected!")
me.infected = 1
print(me)
