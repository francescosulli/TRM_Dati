class Person:

    def __init__(self, firstname, lastname):
        self.first = firstname
        self.last = lastname
        self.fullname = self.first + ' '+ self.last

person = Person('Fra', 'Sulli')
print(person.fullname)
person.first = 'Francesco'
print(person.fullname, person.first)