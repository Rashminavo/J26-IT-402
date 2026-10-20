import 'package:flutter/material.dart';
import '../theme/app_colors.dart';
import 'landing_page.dart';

class ProfilePage extends StatelessWidget {
  const ProfilePage({super.key});

  void _showMessage(BuildContext context, String label) {
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(SnackBar(content: Text('$label is coming soon.'), behavior: SnackBarBehavior.floating));
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(24, 20, 24, 14),
      child: Column(
        children: [
          Stack(
            alignment: Alignment.bottomRight,
            children: [
              Container(
                width: 82,
                height: 82,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: const Color(0xFFD9D7E9),
                  border: Border.all(color: Colors.white, width: 3),
                  boxShadow: [BoxShadow(color: AppColors.ink.withValues(alpha: .12), blurRadius: 10, offset: const Offset(0, 3))],
                ),
                child: const Icon(Icons.person_rounded, size: 48, color: Color(0xFF77758A)),
              ),
              Container(
                width: 23,
                height: 23,
                decoration: BoxDecoration(color: AppColors.purple, shape: BoxShape.circle, border: Border.all(color: Colors.white, width: 2)),
                child: const Icon(Icons.edit_rounded, color: Colors.white, size: 11),
              ),
            ],
          ),
          const SizedBox(height: 10),
          const Text('Kamal Deep', style: TextStyle(color: AppColors.ink, fontSize: 16, fontWeight: FontWeight.w800)),
          const SizedBox(height: 3),
          const Text('kamal@example.com', style: TextStyle(color: Color(0xFF9694AA), fontSize: 10)),
          const SizedBox(height: 17),
          Row(
            children: const [
              Expanded(child: _ProfileStat(value: '24', label: 'SESSIONS')),
              SizedBox(width: 9),
              Expanded(child: _ProfileStat(value: '12', label: 'STREAK')),
              SizedBox(width: 9),
              Expanded(child: _ProfileStat(value: '68', label: 'AVG SCORE')),
            ],
          ),
          const SizedBox(height: 18),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 13),
            decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(13)),
            child: Column(
              children: [
                _ProfileRow(icon: Icons.person_outline_rounded, title: 'Edit Profile', onTap: () => _showMessage(context, 'Edit Profile')),
                _ProfileRow(icon: Icons.notifications_none_rounded, title: 'Notifications', onTap: () => _showMessage(context, 'Notifications')),
                _ProfileRow(icon: Icons.shield_outlined, title: 'Privacy & Data', onTap: () => _showMessage(context, 'Privacy & Data')),
                _ProfileRow(icon: Icons.language_rounded, title: 'Language', trailing: 'English', onTap: () => _showMessage(context, 'Language')),
                _ProfileRow(icon: Icons.help_outline_rounded, title: 'Help & Support', onTap: () => _showMessage(context, 'Help & Support')),
                _ProfileRow(icon: Icons.info_outline_rounded, title: 'About MindSense', isLast: true, onTap: () => _showMessage(context, 'About MindSense')),
              ],
            ),
          ),
          const SizedBox(height: 16),
          SizedBox(
            width: double.infinity,
            height: 44,
            child: OutlinedButton.icon(
              onPressed: () => Navigator.of(context).pushAndRemoveUntil(
                MaterialPageRoute<void>(builder: (_) => const LandingPage()),
                (route) => false,
              ),
              icon: const Icon(Icons.logout_rounded, size: 14),
              label: const Text('Logout', style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600)),
              style: OutlinedButton.styleFrom(
                foregroundColor: const Color(0xFFFF5C6C),
                side: const BorderSide(color: Color(0xFFFF6A78)),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(11)),
              ),
            ),
          ),
          const SizedBox(height: 10),
          const Text('Version 1.2.0 (Build 452)', style: TextStyle(color: Color(0xFFA5A3B4), fontSize: 9)),
        ],
      ),
    );
  }
}
class _ProfileStat extends StatelessWidget {
  const _ProfileStat({required this.value, required this.label});

  final String value;
  final String label;

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(vertical: 11),
        decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(11)),
        child: Column(children: [Text(value, style: const TextStyle(color: AppColors.purple, fontSize: 16, fontWeight: FontWeight.w800)), const SizedBox(height: 3), Text(label, style: const TextStyle(color: Color(0xFF9998A8), fontSize: 8, fontWeight: FontWeight.w600))]),
      );
}

class _ProfileRow extends StatelessWidget {
  const _ProfileRow({required this.icon, required this.title, required this.onTap, this.trailing, this.isLast = false});

  final IconData icon;
  final String title;
  final String? trailing;
  final VoidCallback onTap;
  final bool isLast;

  @override
  Widget build(BuildContext context) => InkWell(
        onTap: onTap,
        child: Container(
          height: 47,
          decoration: BoxDecoration(border: isLast ? null : const Border(bottom: BorderSide(color: Color(0xFFF1F0F6)))),
          child: Row(
            children: [
              Container(width: 26, height: 26, decoration: const BoxDecoration(color: Color(0xFFF0EEFF), shape: BoxShape.circle), child: Icon(icon, color: AppColors.purple, size: 14)),
              const SizedBox(width: 10),
              Expanded(child: Text(title, style: const TextStyle(color: AppColors.ink, fontSize: 11, fontWeight: FontWeight.w500))),
              if (trailing != null) Text(trailing!, style: const TextStyle(color: Color(0xFF9998A8), fontSize: 9)),
              const SizedBox(width: 6),
              const Icon(Icons.chevron_right_rounded, color: Color(0xFFB6B4C4), size: 17),
            ],
          ),
        ),
      );
}
