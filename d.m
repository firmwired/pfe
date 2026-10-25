ed = 2.71828;
y = -0.9 : 0.01 : 2;

% Added '.' before '*' and before '^'
c = 3 * y .* log(y) - (1 / 36) * exp(-(36 * y - (36 / ed)).^4);

% Using real(c) to safely handle the negative y values in log(y)
plot(y, real(c), "b--", "LineWidth", 2)
grid on;
xlabel("y");
ylabel("c");

