import 'package:flutter/material.dart';
import '../theme/app_colors.dart';
import 'login_page.dart';

class LandingPage extends StatelessWidget {
  const LandingPage({super.key});

  static const purple = AppColors.purple;
  static const ink = AppColors.ink;

  void _showMessage(BuildContext context, String message) {
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(SnackBar(
        content: Text(message),
        behavior: SnackBarBehavior.floating,
        backgroundColor: ink,
        duration: const Duration(seconds: 2),
      ));
  }

  @override
  Widget build(BuildContext context) {
    final height = MediaQuery.sizeOf(context).height;
    final compact = height < 700;
    return Scaffold(
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 480),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: Column(
                children: [
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      Container(
                        width: 34,
                        height: 34,
                        decoration: BoxDecoration(
                          color: const Color(0xFFF0EEFF),
                          borderRadius: BorderRadius.circular(11),
                        ),
                        child: const Icon(Icons.spa_rounded, color: purple, size: 20),
                      ),
                      const SizedBox(width: 9),
                      const Text('MindSense', style: TextStyle(color: ink, fontWeight: FontWeight.w800, fontSize: 16)),
                    ],
                  ),
                  Expanded(
                    child: SingleChildScrollView(
                      child: Column(
                        children: [
                          SizedBox(height: compact ? 22 : 38),
                          const _MindfulnessOrb(),
                          SizedBox(height: compact ? 19 : 28),
                          const Text(
                            'Your AI Mental Health\nCompanion',
                            textAlign: TextAlign.center,
                            style: TextStyle(color: ink, fontSize: 23, height: 1.16, fontWeight: FontWeight.w800, letterSpacing: -0.5),
                          ),
                          const SizedBox(height: 10),
                          const Text(
                            'Detect stress, anxiety and depression\nearly with AI.',
                            textAlign: TextAlign.center,
                            style: TextStyle(color: Color(0xFF5E5A89), fontSize: 13, height: 1.5),
                          ),
                          SizedBox(height: compact ? 20 : 30),
                          SizedBox(
                            width: double.infinity,
                            height: 48,
                            child: FilledButton(
                              onPressed: () => Navigator.of(context).push(
                                MaterialPageRoute<void>(builder: (_) => const LoginPage()),
                              ),
                              style: FilledButton.styleFrom(
                                backgroundColor: purple,
                                foregroundColor: Colors.white,
                                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                                elevation: 5,
                                shadowColor: purple.withValues(alpha: 0.28),
                              ),
                              child: const Row(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [Text('Get Started', style: TextStyle(fontWeight: FontWeight.w700, fontSize: 14)), SizedBox(width: 8), Icon(Icons.arrow_forward_rounded, size: 17)],
                              ),
                            ),
                          ),
                          const SizedBox(height: 11),
                          SizedBox(
                            width: double.infinity,
                            height: 46,
                            child: OutlinedButton(
                              onPressed: () => _showMessage(context, 'Learn more about your wellbeing journey.'),
                              style: OutlinedButton.styleFrom(
                                foregroundColor: purple,
                                side: const BorderSide(color: purple, width: 1.2),
                                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                              ),
                              child: const Text('Learn More', style: TextStyle(fontWeight: FontWeight.w600, fontSize: 14)),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  Container(
                    margin: const EdgeInsets.only(top: 12, bottom: 14),
                    padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 8),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(18),
                      border: Border.all(color: const Color(0xFFF2F1FA)),
                      boxShadow: [BoxShadow(color: ink.withValues(alpha: 0.035), blurRadius: 18, offset: const Offset(0, 5))],
                    ),
                    child: const Row(
                      mainAxisAlignment: MainAxisAlignment.spaceAround,
                      children: [
                        _FeatureItem(icon: Icons.chat_bubble_outline_rounded, label: 'AI Chatbot'),
                        _FeatureItem(icon: Icons.auto_graph_rounded, label: 'Analytics'),
                        _FeatureItem(icon: Icons.verified_user_outlined, label: 'Privacy Safe'),
                      ],
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
}

class _MindfulnessOrb extends StatelessWidget {
  const _MindfulnessOrb();

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.sizeOf(context).width.clamp(230.0, 330.0) * 0.82;
    return SizedBox(
      width: size,
      height: size,
      child: Stack(
        alignment: Alignment.center,
        children: [
          Container(
            width: size,
            height: size,
            decoration: BoxDecoration(shape: BoxShape.circle, color: const Color(0xFFEDEBFF).withValues(alpha: .68)),
          ),
          Container(
            width: size * .79,
            height: size * .79,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: const Color(0xFFDCD8FA),
              boxShadow: [BoxShadow(color: const Color(0xFF8E83E5).withValues(alpha: .16), blurRadius: 24, spreadRadius: 1)],
            ),
          ),
          Container(
            width: size * .66,
            height: size * .66,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              gradient: const RadialGradient(colors: [Color(0xFFC9C5F2), Color(0xFFD1CDF5)], center: Alignment(-.22, -.25), radius: .9),
              border: Border.all(color: Colors.white.withValues(alpha: .3), width: 1),
            ),
          ),
        ],
      ),
    );
  }
}

class _FeatureItem extends StatelessWidget {
  const _FeatureItem({required this.icon, required this.label});

  final IconData icon;
  final String label;

  @override
  Widget build(BuildContext context) => Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 42,
            height: 42,
            decoration: BoxDecoration(color: const Color(0xFFF0EEFF), borderRadius: BorderRadius.circular(12)),
            child: Icon(icon, color: AppColors.purple, size: 20),
          ),
          const SizedBox(height: 7),
          Text(label, style: const TextStyle(color: AppColors.ink, fontSize: 10, fontWeight: FontWeight.w600)),
        ],
      );
}
