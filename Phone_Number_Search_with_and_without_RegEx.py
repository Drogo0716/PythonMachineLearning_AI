import re

# Without Reg-Ex
def isPhoneNumber(text):
    if len(text) != 12:
        return False
    for i in range(0,3):
        if not text[i].isdecimal():
            return False
    if text[3] != '-':
        return False
    for i in range(4,7):
        if not text[i].isdecimal():
            return False
    if text[7] != '-':
        return False
    for i in range(8,12):
        if not text[i].isdecimal():
            return False
    return True


print('415-555-4242 is a phone number:')
print(isPhoneNumber('415-555-4242'))
print('Moshi moshi is a phone number:')
print(isPhoneNumber('Moshi moshi'))

message = "Call me at 903-690-2748 tomorrow. 903-819-9090 is my office."
for i in range(len(message)):
    chunk = message[i:i+12]
    if isPhoneNumber(chunk):
        print('Phone number found: ' + chunk)
print('Done')

# With Reg-Ex
phoneNumRegex = re.compile(r'(\d{3})-(\d{3}-\d{4})')
result = phoneNumRegex.search(message)
print('Phone Number found: ' + result.group())
print(result.group(1))
print(result.group(2))
print(result.group(0) + '\n')

# Notice the diference in the groups() method compared to group()
areaCode, phoneNumber = result.groups()

print(areaCode)
print(phoneNumber)

# What if you need parentheses around the area code?
# phoneNumRegex = re.compile(r'(\(d{3}\))-(\d{3}-\d{4})')
# result2 = phoneNumRegex.search(message)
# areaCode, phoneNumber = result.groups()
# print(result.group(1))


# Use the pipe to match one of several patterns within a group, and findall() instead of search() to return string
# list of all matches (Really should NOT use findall() when groups are present in regex)
batRegex = re.compile(r'Bat(man|mobile|copter|bat)')
result = batRegex.findall('Batmobile lost a wheel, and the Joker got away in the Batcopter while swinging the Batbat.')
print(result)
# Optional Matching with the Question Mark
#batRegex = re.compile(r'Bat(wo)?man')
#result = batRegex.search('The Adventures of Batman')
#print(result.group())
#result = batRegex.search('The Adventures of Batwoman')
#print(result.group())

phoneNumRegex = re.compile(r'\d\d\d-\d\d\d-\d\d\d\d')
result = phoneNumRegex.findall('My cell phone number is 903-690-2748, and my office phone number is 903-555-1234')
print(result)

# Findall() with groups
phoneNumRegex = re.compile(r'(\d\d\d)-(\d\d\d)-(\d\d\d\d)') # has groups
result = phoneNumRegex.findall('My cell phone number is 903-690-2748, and my office phone number is 903-555-1234')
print(result)


# making own char classes
vowelRegex = re.compile(r'[aeiouAEIOU]')
result = vowelRegex.findall('RoboCop eats baby food. BABY FOOD.')
print(result)

# Negative character class with caret character
vowelRegex = re.compile(r'[^aeiouAEIOU]')
result = vowelRegex.findall('RoboCop eats baby food. BABY FOOD.')
print(result)

