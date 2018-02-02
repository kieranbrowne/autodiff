(ns autodiff.core-test
  (:refer-clojure :exclude  [* + - / identity])
  ;; (:import [autodiff.core Dual])
  (:require [clojure.test :refer :all]
            [autodiff.core :refer :all]
            [autodiff.protocols :refer :all]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as mat]
            [clojure-tensorflow.ops :as tf]
            [clojure-tensorflow.core :refer [run]]

            ;; [autodiff.protocols :as ad]
            ))



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
    (is (= 1 (d add 3 (->Dual 2 1)))))
  (testing "derivative of add (assumes both are variable)"
    (is (= 2 (d add (->Dual 3 1) (->Dual 2 1)))))
  )

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


(defn approx= [a b]
  (and
    (< a (clojure.core/+ b 0.0001))
    (> a (clojure.core/- b 0.0001))))

;; Needed ops
(deftest tf-ops
  (let []

    (testing "Const op"
      (is (= 0
           (d constant 1)))
      (is (= 0
             (d add (constant 1) (constant 1))))
      (is (= 0
             (d add (->Dual (constant 1) 0) (constant 1))))
      (is (= 0
             (d add (constant 1) (->Dual (constant 1) 0))))
      )

    (testing "Add op"
      (let [x (constant 3)
            y (constant -2)]
        (is (= 1 (add x y)))
        (is (= 1 (add y x)))
        (is (= 2 (d add (wrt y) (wrt x))))
        (is (= 1 (d add x (wrt y))))
        (is (= 1 (d add (wrt x) y)))
        ))

    (testing "Sub op"
      (let [x (constant 3)
            y (constant -2)]
        (is (= 5 (sub x y)))
        (is (= -5 (sub y x)))
        (is (= 0 (d sub y x)))
        (is (= -1 (d sub x (wrt y))))
        (is (= 1 (d sub (wrt x) y)))
        ))

    (testing "Mul op"
      (let [x (constant 3)
            y (constant -2)]
        (is (= -6 (mul x y)))
        (is (= -6 (mul y x)))
        (is (= 0 (d mul y x))) ; both are constant
        (is (= 3 (d mul x (wrt y))))
        (is (= -2 (d mul (wrt x) y)))
        ))

    (testing "Div op"
      (let [x (constant 3)
            y (constant -2)]
        (is (= -3/2 (div x y)))
        (is (approx= -0.66666 (div y x)))
        (is (= 5/9 (d div (wrt y) (wrt x))))
        (is (= -3/4 (d div x (wrt y))))
        (is (= -1/2 (d div (wrt x) y)))
        ))

    (testing "Pow op"
      (let [x (constant 3)
            y (constant 2)]
        (is (approx= 9.0 (pow x y)))
        (is (approx= 8.0 (pow y x)))
        (is (approx= 6.0 (d pow (wrt x) y)))
        (is (approx= 9.88751 (d pow x (wrt y))))))

    (testing "Log op"
      (let [x (coerce 3 1.2)]
        (is (= x (identity x)))
        ))

    (testing "Tanh op"
      (let [x (constant 3.)
            y (constant -2.)]
        (is (approx= 0.99505472 (tanh x)))
        (is (approx= -0.96402758 (tanh y)))
        (is (approx= 0.0098661184 (d tanh (wrt x))))
        (is (approx= 0.070650816 (d tanh (wrt y))))
        ))

    (testing "Sigmoid op"
      (let [x (constant 3.)
            y (constant -2.)]
        (is (approx= 0.95257413 (sigmoid x)))
        (is (approx= 0.11920292 (sigmoid y)))
        (is (approx= 0.045176655 (d sigmoid (wrt x))))
        (is (approx= 0.10499358 (d sigmoid (wrt y))))
        ))

    (testing "Identity op"
      (let [x (constant 3.)
            y (constant -2.)]
        (is (approx= 0.95257413 (sigmoid x)))
        (is (approx= 0.11920292 (sigmoid y)))
        (is (approx= 0.045176655 (d sigmoid (wrt x))))
        (is (approx= 0.10499358 (d sigmoid (wrt y))))
        ))
    ))

(comment
  (let [a [[2 0] [0 2]]
        b [[1 2] [3 4]]]

    (d matmul (wrt a) b)

    ))

(comment
  (let [a [[2. 0. 1.] [0. 2. 1.]]
        b [[1. 2. 3. 4.] [1. 2. 3. 4.] [1. 2. 3. 4.]]]

    (matmul a b)

    (d matmul (wrt a) b)


    ;; (transpose
    ;;  (mul (transpose b)
    ;;       (transpose
    ;;        (sum
    ;;         (transpose
    ;;          (:f' (wrt a)))))))

    ;; (mul
    ;;  (sum (transpose a))
    ;;  (transpose b))

    ;; a
    ;; (wrt a)
    (d matmul a (wrt b))
    ))

;; core.matrix
(deftest core-matrix-grads
  (extend-types
   [clojure.lang.PersistentVector]

   AutoDiff

   (constant [a] a)
   (add [a b]
        (if (not (or (dual? a) (dual? b)))
          (mat/+ a b)
          (add (coerce a) (coerce b))))
   (mul [a b]
        (if (not (or (dual? a) (dual? b)))
          (mat/* a b)
          (mul (coerce a) (coerce b))))
   (matmul [a b]
        (if (and (m/array? a) (m/array? b))
          (m/mmul a b)
          (matmul (coerce a) (coerce b))))
   (transpose [a] (m/transpose a))
   (shape [x])
   ;; (sub [a b]
   ;;      (if (and (number? a) (number? b))
   ;;        (clojure.core/- a b)
   ;;        (ad/sub (ad/coerce a) (ad/coerce b))))
   ;; (mul [a b]
   ;;      (if (and (number? a) (number? b))
   ;;        (clojure.core/* a b)
   ;;        (ad/mul (ad/coerce a) (ad/coerce b))))
   ;; (div [a b]
   ;;      (if (and (number? a) (number? b))
   ;;        (clojure.core// a b)
   ;;        (ad/mul (ad/coerce a) (ad/coerce b))))
   ;; (pow [a b]
   ;;      (if (and (number? a) (number? b))
   ;;        (Math/pow a b)
   ;;        (ad/pow (ad/coerce a) (ad/coerce b))))
   ;; (log [a] (Math/log a))
   ;; (tanh [a] (Math/tanh a))
   ;; (exp [a] (Math/exp a))
   ;; (sigmoid [a] (ad/div 1 (ad/add 1 (ad/pow (ad/exp 1) (ad/negate a)))))
   ;; (negate [a] (clojure.core/- a))
   ;; (sin [a] (Math/sin a))
   ;; (cos [a] (Math/cos a))
   ;; (pi [a] Math/PI)
   (sum [a] (vec (map (partial reduce +) a)))
   (one [a] (m/to-nested-vectors (m/fill (m/new-array (m/shape a)) 1)))
   (zero [a] (m/zero-array (m/shape a)))
   (val-like [a v] (m/to-nested-vectors (m/fill (m/new-array (m/shape a)) v)))
   ;; (two [a] 2.)
   )

  (let [a [[2. 0.] [0. 2.]]
        b [[1. 2.] [3. 4.]]]

    (testing "Basics"
      (is (= [[2. 4.] [6. 8.]] (matmul a b)))

      (is (= [[3. 7.] [3. 7.]]
             (d matmul (wrt a) b)))
      (is (= [[2. 2.] [2. 2.]]
             (d matmul (coerce a (zero a))
                (coerce b (one b)))))
      ))

  (let [a [[2. 0. 1.] [0. 2. 1.]]
        b [[1. 2. 3. 4.] [1. 2. 3. 4.] [1. 2. 3. 4.]]]

    (testing "Basics"
      (is (= [[3. 6. 9. 12.] [3. 6. 9. 12.]] (matmul a b)))

      (is (= [[10. 10. 10.] [10. 10. 10.]]
             (d matmul (wrt a) b)))
      (is (= [[2. 2. 2. 2.] [2. 2. 2. 2.] [2. 2. 2. 2.]]
             (d matmul a (wrt b))))
      ))
  )

(#{1 2} 0)

(defn is-tf? [x]
  (#{org.tensorflow.Operation
     org.tensorflow.Output}
   (type x)))

;; core.matrix
(deftest clojure-tensorflow-grads
  (extend-types
   [org.tensorflow.Operation
    org.tensorflow.Output]

   AutoDiff

   (constant [a] a)
   (add [a b]
        (if (not (or (dual? a) (dual? b)))
          (tf/add a b)
          (add (coerce a) (coerce b))))
   (val-like [a v] (tf/constant v))
   ; (mul [a b]
   ;      (if (not (or (dual? a) (dual? b)))
   ;        (mat/* a b)
   ;        (mul (coerce a) (coerce b))))
   (matmul [a b]
           (if (and (is-tf? a) (is-tf? b))
          (tf/matmul a b)
          (matmul (coerce a) (coerce b))))
   ; (transpose [a] (m/transpose a))
   (one [a] (tf/add (zero a) (tf/constant 1.)))
   (zero [a] (tf/mult a (tf/constant 0.)))
   ;; (two [a] 2.)
   )

  (let [a (tf/constant [[2. 0.] [0. 2.]])
        b (tf/constant [[1. 2.] [3. 4.]])]

    (testing "Basics"
      (is (= [[3. 2.] [3. 6.]] (run (add a b))))
      (is (= 0 (run (d add a b)))) ;; a and b are constant
      (is (= 1 (run (d add (wrt a) b)))) ; add with respect to a
      (is (= 1 (run (d add a (wrt b))))) ; add with respect to a
      (is (= 2 (run (d add (wrt a) (wrt b))))) ; add with respect to a and b
      ))

  (let [a (tf/constant [[2. 0. 1.] [0. 2. 1.]])
        b (tf/constant [[1. 2. 3. 4.] [1. 2. 3. 4.] [1. 2. 3. 4.]])]

    (testing "Matrix multiplication"
      (is (= [[3. 6. 9. 12.] [3. 6. 9. 12.]] (run (matmul a b))))

      (is (= 1
             (run (:f' (wrt a)))))

      (is (= [[10. 10. 10.] [10. 10. 10.]]
             (run (d matmul (->Dual a (one a)) b))))
      (is (= [[2. 2. 2. 2.] [2. 2. 2. 2.] [2. 2. 2. 2.]]
             (run (d matmul a (wrt b)))))
      ))
  )
