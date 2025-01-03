package demo

import "fmt"

func FunDemo1() {
	fmt.Println("hello world")

	var stockcode = 123
	var enddate = "2024-12-31"
	var url = "Code=%d&endDate=%s"
	var target_url = fmt.Sprintf(url, stockcode, enddate)
	fmt.Println(target_url)

	fmt.Printf("%q\n", "\t")

	var r1, r2 int
	r1 = 1
	r2 = r1
	r1 = 2
	fmt.Println(r1)
	fmt.Println(r2)

	modifyValue(&r2)
	fmt.Println(r2)

}

func modifyValue(ptr *int) {
	*ptr = 100
}
