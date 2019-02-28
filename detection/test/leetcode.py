
def isPalindrome( x):
    """
    :type x: int
    :rtype: bool

    """
    ri = 0
    if x < 0:
        return False
    else:
        ri = int(''.join(reversed(str(x))))

    if ri == x:
        return True
    else:
        return False


print (isPalindrome(-121))