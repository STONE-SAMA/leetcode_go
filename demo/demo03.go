package demo

import "fmt"

func FunDemo03() {
	var a int = 100
	var ptr *int

	ptr = &a

	fmt.Printf("指针的地址%x\n", ptr)

	fmt.Printf("指针的值%d\n", *ptr)
}
