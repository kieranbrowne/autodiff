(ns autodiff.core-test
  (:refer-clojure :exclude  [* + - /])
  ;; (:import [autodiff.core Dual])
  (:require [clojure.test :refer :all]
            [autodiff.core :refer :all]))


(deftest basic
  (testing "Simple quadratic"
    (is (= 6 (:f' (#(* % %) (autodiff.core.Dual. 3 1)))))))


(deftest quadratics
  (let [f ; f(x) = 4x^2 + 3
        (fn [x] (+ (* (* x x) 4) 3))
        g ; g(x) = -2x^3 - 2
        (fn [x] (+ -2 (* -2 (exp x 3))))
        ]

    (testing "f(x) = 4x^2 + 3 where x = 3"
      (is (= (new autodiff.core.Dual 39 24)
           (f (autodiff.core.Dual. 3 1)))))

    (testing "f(x) = 4x^2 + 3 where x = 0"
      (is (= (new autodiff.core.Dual 3 0)
             (f (autodiff.core.Dual. 0 1)))))

    (testing "f(x) = 4x^2 + 3 where x = -1"
      (is (= (new autodiff.core.Dual 7 -8)
             (f (autodiff.core.Dual. -1 1)))))

    (testing "g(x) = -2x^3 - 2 where x = 2"
      (is (= (new autodiff.core.Dual -18 -24)
             (g (autodiff.core.Dual. 2 1)))))

    ))
