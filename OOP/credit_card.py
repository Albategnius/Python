class credit:
  """ consumer card """
  def __init__(self, customer, bank, acnt, limit):
    """ card instance """
    self.customer = customer
    self.bank = bank
    self.acnt = acnt
    self.limit = limit
    self.balance = 0
   
  def get_customer(self):
    return self.customer
  def get_bank(self):
    return self.bank
  def get_acnt(self):
    return self.acnt
  def get_limit(self):
    return self.limit
  def get_balance(self):
    return self.balance
  
  def charge(sefl,price):
    """ Charge given price to the card, assuming sufficient credit limit """
    if price + self.balance > self.limit:
      return False
    else:
      self.balance += price
      return True
    
    def make_payment(self,amount):
      self.balance -= amount
    
