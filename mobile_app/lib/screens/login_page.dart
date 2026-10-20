import 'package:flutter/material.dart';
import '../theme/app_colors.dart';
import 'dashboard_page.dart';

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  bool _isLogin = true;
  bool _obscurePassword = true;
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  void _message(String text) {
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(SnackBar(content: Text(text), behavior: SnackBarBehavior.floating));
  }

  InputDecoration _fieldDecoration(String hint, IconData icon, {Widget? suffix}) => InputDecoration(
        hintText: hint,
        prefixIcon: Icon(icon, size: 18, color: const Color(0xFF85849A)),
        suffixIcon: suffix,
        filled: true,
        fillColor: const Color(0xFFF7F7FA),
        hintStyle: const TextStyle(color: Color(0xFFAAA9B5), fontSize: 14),
        contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 15),
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(13), borderSide: BorderSide.none),
        enabledBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(13), borderSide: const BorderSide(color: Color(0xFFF0EFF5))),
        focusedBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(13), borderSide: const BorderSide(color: AppColors.purple)),
      );

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 480),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SizedBox(height: 20),
                  IconButton(
                    onPressed: () => Navigator.of(context).maybePop(),
                    icon: const Icon(Icons.arrow_back_rounded, size: 20),
                    style: IconButton.styleFrom(
                      foregroundColor: AppColors.ink,
                      backgroundColor: Colors.white,
                      side: const BorderSide(color: Color(0xFFF0EFF5)),
                      shape: const CircleBorder(),
                    ),
                  ),
                  const SizedBox(height: 24),
                  Text(_isLogin ? 'Welcome Back' : 'Create Account', style: const TextStyle(color: AppColors.ink, fontSize: 24, fontWeight: FontWeight.w800, letterSpacing: -.4)),
                  const SizedBox(height: 6),
                  Text(_isLogin ? 'Sign in to continue your wellness journey.' : 'Join MindSense and start your wellness journey.', style: const TextStyle(color: Color(0xFF85849A), fontSize: 13)),
                  const SizedBox(height: 24),
                  Row(
                    children: [
                      _tab('Login', true),
                      _tab('Register', false),
                    ],
                  ),
                  const SizedBox(height: 22),
                  const Text('Email Address', style: TextStyle(color: AppColors.ink, fontSize: 12, fontWeight: FontWeight.w700)),
                  const SizedBox(height: 8),
                  TextField(
                    controller: _emailController,
                    keyboardType: TextInputType.emailAddress,
                    decoration: _fieldDecoration('name@example.com', Icons.mail_outline_rounded),
                  ),
                  const SizedBox(height: 18),
                  const Text('Password', style: TextStyle(color: AppColors.ink, fontSize: 12, fontWeight: FontWeight.w700)),
                  const SizedBox(height: 8),
                  TextField(
                    controller: _passwordController,
                    obscureText: _obscurePassword,
                    decoration: _fieldDecoration(
                      'Enter your password',
                      Icons.lock_outline_rounded,
                      suffix: IconButton(
                        onPressed: () => setState(() => _obscurePassword = !_obscurePassword),
                        icon: Icon(_obscurePassword ? Icons.visibility_off_outlined : Icons.visibility_outlined, size: 18, color: const Color(0xFF85849A)),
                      ),
                    ),
                  ),
                  if (_isLogin) ...[
                    Align(
                      alignment: Alignment.centerRight,
                      child: TextButton(
                        onPressed: () => _message('Password reset is not set up yet.'),
                        child: const Text('Forgot password?', style: TextStyle(color: AppColors.purple, fontSize: 12, fontWeight: FontWeight.w500)),
                      ),
                    ),
                  ] else ...[
                    const SizedBox(height: 14),
                  ],
                  SizedBox(
                    width: double.infinity,
                    height: 49,
                    child: FilledButton(
                      onPressed: () {
                        if (_isLogin) {
                          Navigator.of(context).pushAndRemoveUntil(
                            MaterialPageRoute<void>(builder: (_) => const DashboardPage()),
                            (route) => false,
                          );
                        } else {
                          _message('Registration is not connected yet.');
                        }
                      },
                      style: FilledButton.styleFrom(
                        backgroundColor: AppColors.purple,
                        foregroundColor: Colors.white,
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                        elevation: 4,
                        shadowColor: AppColors.purple.withValues(alpha: .28),
                      ),
                      child: Text(_isLogin ? 'Login' : 'Create Account', style: const TextStyle(fontWeight: FontWeight.w700)),
                    ),
                  ),
                  const SizedBox(height: 22),
                  const Row(children: [Expanded(child: Divider(color: Color(0xFFF0EFF5))), Padding(padding: EdgeInsets.symmetric(horizontal: 12), child: Text('or continue with', style: TextStyle(color: Color(0xFF9998A8), fontSize: 11))), Expanded(child: Divider(color: Color(0xFFF0EFF5)))]),
                  const SizedBox(height: 18),
                  SizedBox(
                    width: double.infinity,
                    height: 48,
                    child: OutlinedButton.icon(
                      onPressed: () => _message('Google sign-in is not connected yet.'),
                      icon: const Text('G', style: TextStyle(fontWeight: FontWeight.w800, fontSize: 16, color: Color(0xFF4285F4))),
                      label: const Text('Sign in with Google', style: TextStyle(color: AppColors.ink, fontWeight: FontWeight.w600, fontSize: 13)),
                      style: OutlinedButton.styleFrom(side: const BorderSide(color: Color(0xFFE8E7EF)), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12))),
                    ),
                  ),
                  const Spacer(),
                  Center(
                    child: Padding(
                      padding: const EdgeInsets.only(bottom: 22),
                      child: Wrap(
                        alignment: WrapAlignment.center,
                        children: [
                          const Text('By continuing you agree to our ', style: TextStyle(color: Color(0xFF9998A8), fontSize: 11)),
                          GestureDetector(onTap: () => _message('Terms of Service'), child: const Text('Terms of Service', style: TextStyle(color: AppColors.ink, fontSize: 11, decoration: TextDecoration.underline))),
                          const Text(' & ', style: TextStyle(color: Color(0xFF9998A8), fontSize: 11)),
                          GestureDetector(onTap: () => _message('Privacy Policy'), child: const Text('Privacy Policy', style: TextStyle(color: AppColors.ink, fontSize: 11, decoration: TextDecoration.underline))),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _tab(String label, bool loginTab) {
    final selected = _isLogin == loginTab;
    return Expanded(
      child: InkWell(
        onTap: () => setState(() => _isLogin = loginTab),
        child: Container(
          padding: const EdgeInsets.only(bottom: 11),
          decoration: BoxDecoration(border: Border(bottom: BorderSide(color: selected ? AppColors.purple : const Color(0xFFF0EFF5), width: selected ? 2 : 1))),
          child: Center(child: Text(label, style: TextStyle(color: selected ? AppColors.purple : const Color(0xFF85849A), fontSize: 14, fontWeight: selected ? FontWeight.w600 : FontWeight.w500))),
        ),
      ),
    );
  }
}
