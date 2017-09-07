(ns autodiff.core-test
  (:refer-clojure :exclude  [* + - /])
  ;; (:import [autodiff.core Dual])
  (:require [clojure.test :refer :all]
            [autodiff.core :refer :all]
            [autodiff.protocols :refer :all]

            [autodiff.protocols :as ad]))




(deftest basic
  (testing "Simple quadratic"
    (is (= 6 (:f' (#(* % %) (->Dual 3 1)))))))


(deftest associativity

  (testing "2 * (Dual 6 1) = (Dual 6 1) * 2"
    (is (= (* (->Dual 6 1) 2)
           (* 2 (->Dual 6 1))
           )))

  (testing "> 2 arguments"
    (is (= (+ 4 (->Dual 6 1) 2)
           (+ 2 4 (->Dual 6 1))
           (+ (->Dual 6 1) 2 4)
           (->Dual 12 1)
           )))
  )

(deftest ops
  (testing "derivative of constant"
    (is (= 0 (d constant 3))))
  (testing "derivative of add where one is variable"
    (is (= 1 (d add 3 (->Dual 2 0)))))
  (testing "derivative of add (assumes both are variable)"
    (is (= 2 (d add 3 2))))
  )
(d add 3 (->Dual 2 0))

(deftest quadratics
  (let [f ; f(x) = 4x^2 + 3
        (fn [x] (+ (* (* x x) 4) 3))
        g ; g(x) = -2x^3 - 2
        (fn [x] (+ -2 (* -2 (reduce * (repeat 3 x)))))
        ]

    (testing "f(x) = 4x^2 + 3 where x = 3"
      (is (= (->Dual 39 24)
           (f (->Dual 3 1)))))

    (testing "f(x) = 4x^2 + 3 where x = 0"
      (is (= (->Dual 3 0)
             (f (->Dual 0 1)))))

    (testing "f(x) = 4x^2 + 3 where x = -1"
      (is (= (->Dual 7 -8)
             (f (->Dual -1 1)))))

    (testing "g(x) = -2x^3 - 2 where x = 2"
      (is (= (->Dual -18 -24)
             (g (->Dual 2 1)))))

    (testing "g(x) = -2x^3 - 2 where x = -2"
      (is (= (->Dual 14 -24)
             (g (->Dual -2 1)))))

    (testing "g(x) = -2x^3 - 2 where x = 0"
      (is (= (->Dual -2 0)
             (g (->Dual 0 1)))))

    (testing "(f.g)(x) = f comp g where x = 1"
      (is (= (->Dual 67 192)
             ((comp f g) (->Dual 1 1)))))

    (testing "(f.g)(x) = f comp g where x = -2"
      (is (= (->Dual 787 -2688)
             ((comp f g) (->Dual -2 1)))))

    (testing "(f.g)(x) = f comp g where x = 0.5"
      (is (= (->Dual 23.25 27.0) ((comp f g) (->Dual 0.5 1)))))
    ))
