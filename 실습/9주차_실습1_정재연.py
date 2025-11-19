from layer_naive import *

apple=200
apple_num = 3
apple_tax=1.05

snack=200
snack_num=2
snack_tax=1.1

#계층들
mul_apple_layer = MulLayer()
mul_apple_tax_layer = MulLayer()

mul_snack_layer = MulLayer()
mul_snack_tax_layer = MulLayer()

add_apple_snack_layer = AddLayer()


#순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
apple_tax_price = mul_apple_tax_layer.forward(apple_price, apple_tax)

snack_price = mul_snack_layer.forward(snack, snack_num)
snack_tax_price = mul_snack_tax_layer.forward(snack_price, snack_tax)

price = add_apple_snack_layer.forward(apple_tax_price, snack_tax_price)

# 역전파
dprice = 1
apple_price, apple_tax = mul_apple_tax_layer.backward(dprice)
apple, apple_num = mul_apple_layer.backward(apple_price)

snack_price, snack_tax = mul_snack_tax_layer.backward(dprice)
snack, snack_num = mul_snack_layer.backward(snack_price)

print("price:", int(price))
print("Apple:", apple)
print("Apple_num:", int(apple_num))
print("apple Tax:", apple_tax)

print("Snack:", snack)
print("Snack_num:", int(snack_num))
print("Snack_tax:", snack_tax)

