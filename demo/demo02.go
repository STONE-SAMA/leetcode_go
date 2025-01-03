package demo

import "fmt"

func FunDemo2() {
	num := 2

	switch num {
	case 1:
		fmt.Println("1")
	case 2:
		fmt.Println("2")
		fallthrough
	case 3:
		fmt.Println("3")
	}

	fmt.Println("end of switch")

	var ret = max(10, 15)
	fmt.Println(ret)

}

func max(a int, b int) int {
	if a > b {
		return a
	} else {
		return b
	}
}
